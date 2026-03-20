"""
Run this to inspect the structure of your weights file.
Output tells us exactly how to rebuild the architecture to match.

Usage:
    python inspect_weights.py
"""
import h5py
import os

WEIGHTS = os.path.join("model", "vgg16_weights.weights.h5")

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  DATASET  {name}  shape={obj.shape}  dtype={obj.dtype}")
    else:
        print(f"  GROUP    {name}")

print(f"Inspecting: {WEIGHTS}\n")
with h5py.File(WEIGHTS, "r") as f:
    print("=== Top-level keys ===")
    for k in f.keys():
        print(f"  {k}")
    
    print("\n=== Full structure (first 80 items) ===")
    count = [0]
    def visitor(name, obj):
        if count[0] < 80:
            print_structure(name, obj)
            count[0] += 1
    f.visititems(visitor)