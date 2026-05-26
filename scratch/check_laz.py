import laspy
import numpy as np

path = "data/Ontario/Hamilton/ward_1/raw/ward_1.laz"
print(f"Reading {path}...")
with laspy.open(path) as f:
    header = f.header
    print("Header bounds:")
    print(f"  X: {header.min[0]} to {header.max[0]}")
    print(f"  Y: {header.min[1]} to {header.max[1]}")
    print(f"  Z: {header.min[2]} to {header.max[2]}")
    print(f"  Number of points: {header.point_count}")

# Read points in chunks to find actual min/max
print("Calculating actual min/max of points...")
actual_min_x = float('inf')
actual_max_x = float('-inf')
actual_min_y = float('inf')
actual_max_y = float('-inf')

with laspy.open(path) as f:
    for chunk in f.chunk_iterator(1_000_000):
        x = chunk.x
        y = chunk.y
        actual_min_x = min(actual_min_x, np.min(x))
        actual_max_x = max(actual_max_x, np.max(x))
        actual_min_y = min(actual_min_y, np.min(y))
        actual_max_y = max(actual_max_y, np.max(y))

print("Actual bounds of points:")
print(f"  X: {actual_min_x} to {actual_max_x}")
print(f"  Y: {actual_min_y} to {actual_max_y}")
