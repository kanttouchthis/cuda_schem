import nbtlib
from nbtlib import tag


def write_schem(blocks_dict:dict, filename, data_version=4325):
    xs = [x for x, y, z in blocks_dict]
    ys = [y for x, y, z in blocks_dict]
    zs = [z for x, y, z in blocks_dict]
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