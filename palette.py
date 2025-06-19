import argparse
import json
from glob import glob

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("pattern", type=str)
parser.add_argument("output", type=str)
args = parser.parse_args()

def find_textures(pattern):
    files = glob(pattern)
    names = ["minecraft:"+file.split("\\")[1].split(".")[0] for file in files]
    return files, names

def average_color(filename):
    img = Image.open(filename).convert("RGB")
    avg = np.array(img).mean(0).mean(0).astype(int).tolist()
    return avg

files, names = find_textures(args.pattern)
palette = {}
for file, name in zip(files, names):
    rgb = average_color(file)
    palette[name] = rgb

with open(args.output, "w") as f:
    json.dump(palette, f, indent=4)