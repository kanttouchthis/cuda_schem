import nbtlib

from time import perf_counter_ns
from nbtlib import tag


def write_schem(blocks_dict:dict, filename, data_version=4325):
    xs = [x for x, _, _ in blocks_dict]
    ys = [y for _, y, _ in blocks_dict]
    zs = [z for _, _, z in blocks_dict]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    width  = max_x - min_x + 1
    height = max_y - min_y + 1
    length = max_z - min_z + 1
    size = width * height * length

    norm_blocks = {
        (x - min_x, y - min_y, z - min_z): block
        for (x, y, z), block in blocks_dict.items()
    }

    palette = {"minecraft:air":0}
    for block in sorted(set(norm_blocks.values())):
        palette[block] = len(palette)

    block_indices = [0] * size
    for (x, y, z), block in norm_blocks.items():
        index = x + width * (z + length * y)
        block_indices[index] = palette[block]

    nbt_data = nbtlib.Compound({"Schematic": nbtlib.Compound({
        'Version': tag.Int(3),
        'DataVersion': tag.Int(data_version),
        'Width': tag.Short(width),
        'Height': tag.Short(height),
        'Length': tag.Short(length),
        'Blocks': nbtlib.Compound({
            'BlockEntities': tag.List[nbtlib.Compound]([]),
            'Palette':nbtlib.Compound({name: tag.Int(index) for name, index in palette.items()}),
            'Data':tag.ByteArray(block_indices),
        }
        )
    })})

    nbt_file = nbtlib.File(nbt_data, filename=filename, gzipped=True)
    nbt_file.save()


def write_structure(blocks_dict, filename, data_version=3953):
    xs = [x for x, _, _ in blocks_dict]
    ys = [y for _, y, _ in blocks_dict]
    zs = [z for _, _, z in blocks_dict]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    length = max_z - min_z + 1
    size = nbtlib.List([nbtlib.Int(width), nbtlib.Int(height), nbtlib.Int(length)])

    norm_blocks = {
        (x - min_x, y - min_y, z - min_z): block
        for (x, y, z), block in blocks_dict.items()
    }

    palette = {"minecraft:air": nbtlib.Int(0)}
    for block in sorted(set(norm_blocks.values())):
        palette[block] = nbtlib.Int(len(palette))

    # Precompute a list of NBT lib ints for speedup
    n_ints = [nbtlib.Int(x) for x in range(max([width, height, length]))]

    blocks = []
    t0 = perf_counter_ns()
    # Fill the blocks list with air first, and if there is a block in that spot put it there.
    for x in range(width):
        for y in range(height):
            for z in range(length):
                block = norm_blocks.get((x, y, z))
                if block:
                    palette_id = palette[block]
                else:
                    palette_id = palette.get("minecraft:air")
                blocks.append(
                    nbtlib.Compound({
                        "pos": nbtlib.List([n_ints[x], n_ints[y], n_ints[z]]),
                        "state": palette_id
                    })
                )
    t1 = perf_counter_ns()
    print(f"Schematic Block Fill: {int((t1-t0)/1000000)}ms")

    t0 = perf_counter_ns()
    blocks = nbtlib.List(blocks)
    nbt_data = nbtlib.Compound(
        {
            "blocks": blocks,
            "entities": nbtlib.List(),
            "palette": nbtlib.List(
                [
                    nbtlib.Compound({"Name": nbtlib.String(name)})
                    for name in palette.keys()
                ]
            ),
            "size": size,
            "DataVersion": nbtlib.Int(data_version),
        }
    )
    t1 = perf_counter_ns()
    print(f"Building NBT Data: {int((t1-t0)/1000000)}ms")

    t0 = perf_counter_ns()
    nbt_file = nbtlib.File(nbt_data, filename=filename)
    nbt_file.save()
    t1 = perf_counter_ns()
    print(f"Compressing and Saving NBT Data: {int((t1-t0)/1000000)}ms")
