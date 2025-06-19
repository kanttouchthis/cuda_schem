import argparse
import json
from time import perf_counter_ns

import numba
import numpy as np
import trimesh
from numba import cuda
from PIL import Image
from trimesh.voxel import VoxelGrid

from schem import write_schem


@cuda.jit(device=True)
def triangle_box_intersect_sat(v0, v1, v2, box_center, box_half_size):
    # Translate triangle to box's local space
    tv0 = cuda.local.array(3, dtype=numba.float32)
    tv1 = cuda.local.array(3, dtype=numba.float32)
    tv2 = cuda.local.array(3, dtype=numba.float32)

    for i in range(3):
        tv0[i] = v0[i] - box_center[i]
        tv1[i] = v1[i] - box_center[i]
        tv2[i] = v2[i] - box_center[i]

    # Triangle edges
    e0 = cuda.local.array(3, dtype=numba.float32)
    e1 = cuda.local.array(3, dtype=numba.float32)
    e2 = cuda.local.array(3, dtype=numba.float32)
    for i in range(3):
        e0[i] = tv1[i] - tv0[i]
        e1[i] = tv2[i] - tv1[i]
        e2[i] = tv0[i] - tv2[i]

    # Helper
    def find_min_max(x0, x1, x2):
        return min(x0, x1, x2), max(x0, x1, x2)

    # 1. Test AABB face normals (X, Y, Z)
    for i in range(3):
        min_v, max_v = find_min_max(tv0[i], tv1[i], tv2[i])
        if min_v > box_half_size[i] or max_v < -box_half_size[i]:
            return False

    # 2. Test triangle normal axis
    normal = cuda.local.array(3, dtype=numba.float32)
    normal[0] = e0[1] * e1[2] - e0[2] * e1[1]
    normal[1] = e0[2] * e1[0] - e0[0] * e1[2]
    normal[2] = e0[0] * e1[1] - e0[1] * e1[0]

    d = normal[0] * tv0[0] + normal[1] * tv0[1] + normal[2] * tv0[2]
    r = (
        box_half_size[0] * abs(normal[0])
        + box_half_size[1] * abs(normal[1])
        + box_half_size[2] * abs(normal[2])
    )
    if abs(d) > r:
        return False

    # 3. Test 9 cross product axes (edge × AABB axis)
    def axis_test(edge, a, b, fa, fb, v0, v1, v2, box_half_size):
        p0 = edge[a] * v0[b] - edge[b] * v0[a]
        p1 = edge[a] * v1[b] - edge[b] * v1[a]
        p2 = edge[a] * v2[b] - edge[b] * v2[a]
        min_p = min(p0, p1, p2)
        max_p = max(p0, p1, p2)
        rad = fa * box_half_size[a] + fb * box_half_size[b]
        return not (min_p > rad or max_p < -rad)

    fa0, fa1, fa2 = abs(e0[0]), abs(e0[1]), abs(e0[2])
    if not axis_test(e0, 1, 2, fa1, fa2, tv0, tv1, tv2, box_half_size):
        return False
    if not axis_test(e0, 0, 2, fa0, fa2, tv0, tv1, tv2, box_half_size):
        return False
    if not axis_test(e0, 0, 1, fa0, fa1, tv0, tv1, tv2, box_half_size):
        return False

    fb0, fb1, fb2 = abs(e1[0]), abs(e1[1]), abs(e1[2])
    if not axis_test(e1, 1, 2, fb1, fb2, tv0, tv1, tv2, box_half_size):
        return False
    if not axis_test(e1, 0, 2, fb0, fb2, tv0, tv1, tv2, box_half_size):
        return False
    if not axis_test(e1, 0, 1, fb0, fb1, tv0, tv1, tv2, box_half_size):
        return False

    fc0, fc1, fc2 = abs(e2[0]), abs(e2[1]), abs(e2[2])
    if not axis_test(e2, 1, 2, fc1, fc2, tv0, tv1, tv2, box_half_size):
        return False
    if not axis_test(e2, 0, 2, fc0, fc2, tv0, tv1, tv2, box_half_size):
        return False
    if not axis_test(e2, 0, 1, fc0, fc1, tv0, tv1, tv2, box_half_size):
        return False

    return True


@cuda.jit
def voxelize_kernel(vertices, faces, voxels, voxel_size, grid_origin, grid_dim):
    tri_idx = cuda.grid(1)
    if tri_idx >= faces.shape[0]:
        return

    # Get triangle vertex indices
    i0, i1, i2 = faces[tri_idx]

    # Get triangle vertices
    v0 = vertices[i0]
    v1 = vertices[i1]
    v2 = vertices[i2]

    # Compute triangle AABB
    min_corner = cuda.local.array(3, dtype=numba.float32)
    max_corner = cuda.local.array(3, dtype=numba.float32)

    half = voxel_size * 0.5
    box_half_size = cuda.local.array(3, dtype=numba.float32)
    box_half_size[0] = half
    box_half_size[1] = half
    box_half_size[2] = half

    for i in range(3):
        min_corner[i] = min(v0[i], v1[i], v2[i])
        max_corner[i] = max(v0[i], v1[i], v2[i])

    # Compute voxel AABB indices
    min_idx = cuda.local.array(3, dtype=numba.int32)
    max_idx = cuda.local.array(3, dtype=numba.int32)

    for i in range(3):
        min_idx[i] = max(int((min_corner[i] - grid_origin[i]) / voxel_size), 0)
        max_idx[i] = min(
            int((max_corner[i] - grid_origin[i]) / voxel_size), grid_dim[i] - 1
        )

    # Check each voxel in bounding box
    for x in range(min_idx[0], max_idx[0] + 1):
        for y in range(min_idx[1], max_idx[1] + 1):
            for z in range(min_idx[2], max_idx[2] + 1):
                # Center of voxel
                voxel_center = cuda.local.array(3, dtype=numba.float32)
                voxel_center[0] = grid_origin[0] + (x + 0.5) * voxel_size
                voxel_center[1] = grid_origin[1] + (y + 0.5) * voxel_size
                voxel_center[2] = grid_origin[2] + (z + 0.5) * voxel_size
                if triangle_box_intersect_sat(v0, v1, v2, voxel_center, box_half_size):
                    cuda.atomic.max(voxels, (x, y, z), tri_idx + 1)


def get_colors(mesh, texture_path):
    texture = np.array(Image.open(texture_path).convert("RGB"))  # shape: (H, W, 3)
    texture_h, texture_w = texture.shape[:2]
    uvs = mesh.visual.uv  # shape: (num_vertices, 2)
    faces = mesh.faces  # shape: (num_faces, 3)
    face_uvs = uvs[faces]  # shape: (num_faces, 3, 2)
    face_pixels = face_uvs * [texture_w - 1, texture_h - 1]
    face_pixels[:, :, 1] = texture_h - 1 - face_pixels[:, :, 1]
    face_pixels = np.round(face_pixels).astype(int)
    colors = []
    for triangle in face_pixels:
        pts = triangle.clip([[0, 0]], [[texture_w - 1, texture_h - 1]])
        sampled_colors = texture[pts[:, 1], pts[:, 0]]
        avg_color = sampled_colors.mean(axis=0)
        colors.append(avg_color)
    colors = np.array(colors, dtype=np.uint8)  # shape: (num_faces, 3)
    return colors


def quantize_colors(colors, palette):
    palette = np.array(palette)
    colors = colors[:, np.newaxis, :]  # shape: (N, 1, 3)
    palette = palette[np.newaxis, :, :]  # shape: (1, M, 3)
    distances = np.linalg.norm(colors - palette, axis=2)  # shape: (N, M)
    nearest_indices = np.argmin(distances, axis=1)  # shape: (N,)
    return nearest_indices


def load_mesh_and_voxelize_color(
    model_path,
    texture_path=None,
    palette: dict = {"minecraft:stone": (0, 0, 0)},
    N_voxels=256,
):
    palette_rgb = list(palette.values())

    mesh = trimesh.load(model_path, process=True, force="mesh")
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    if texture_path is not None:
        colors = get_colors(mesh, texture_path)
        qcolors_idx = quantize_colors(colors, palette_rgb)
    else:
        qcolors_idx = np.array([0] * faces.shape[0])

    # Compute grid bounds
    bounds = mesh.bounds
    min_bound = bounds[0]
    max_bound = bounds[1]
    voxel_size = np.max(max_bound - min_bound) / N_voxels
    grid_dim = np.ceil((max_bound - min_bound) / voxel_size).astype(np.int32)

    # Allocate voxel grid
    voxels = np.zeros(tuple(grid_dim), dtype=np.int64)

    # Transfer to GPU
    d_vertices = cuda.to_device(vertices)
    d_faces = cuda.to_device(faces)
    d_voxels = cuda.to_device(voxels)
    d_min_bound = cuda.to_device(min_bound.astype(np.float32))
    d_grid_dim = cuda.to_device(grid_dim)

    # Launch kernel
    threads_per_block = 32
    blocks = (faces.shape[0] + threads_per_block - 1) // threads_per_block
    voxelize_kernel[blocks, threads_per_block](
        d_vertices, d_faces, d_voxels, voxel_size, d_min_bound, d_grid_dim
    )

    # Copy result back
    voxels = d_voxels.copy_to_host()
    blocks_dict = {}
    palette_keys = list(palette.keys())
    nonzero_coords = np.argwhere(voxels)
    voxel_values = voxels[tuple(nonzero_coords.T)] - 1
    colors = qcolors_idx[voxel_values]
    blocks = [palette_keys[color] for color in colors]
    blocks_dict = dict(zip(map(tuple, nonzero_coords), blocks))

    return blocks_dict


def visualize_voxels_trimesh(voxels):
    vg = VoxelGrid(encoding=voxels)
    vg.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to model (ply, obj, stl, etc.)")
    parser.add_argument("output", type=str, help="Path to .schem output")
    parser.add_argument("--texture", "-t", type=str, default=None, help="Path to texture")
    parser.add_argument("--palette", "-p", default=None, help="Path to .json palette")
    parser.add_argument("--N_voxels", "-n", type=int, default=128, help="Maximum number of voxels in each direction (XYZ)")
    args = parser.parse_args()

    if args.palette is not None:
        palette = json.load(open(args.palette, "r"))
    else:
        palette = {"minecraft:stone": (0, 0, 0)}

    t0 = perf_counter_ns()
    voxels = load_mesh_and_voxelize_color(
        args.model, args.texture, palette, args.N_voxels
    )
    t1 = perf_counter_ns()
    print(f"Voxelization: {int((t1-t0)/1000000)}ms")
    t0 = perf_counter_ns()
    write_schem(voxels, args.output)
    t1 = perf_counter_ns()
    print(f"Schem export: {int((t1-t0)/1000000)}ms")