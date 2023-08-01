#import open3d as o3d
#import os
#
#print(os.getcwd())
#
#print("specify name to open the pcd file")
#path = input()
#pcd = o3d.io.read_point_cloud(f"{path}.pcd")
#o3d.visualization.draw_geometries([pcd])
#
#print('pcd_color', pcd.colors)

#import open3d as o3d
#import os
#import numpy as np
#
#print(os.getcwd())
#
#print("specify name to open the pcd file")
#path = input()
#pcd = o3d.io.read_point_cloud(f"{path}.pcd")
#o3d.visualization.draw_geometries([pcd])
#
#print('pcd_color', pcd.colors)
#
## Convert the point cloud to a numpy array
#points = np.asarray(pcd.points)
#colors = np.asarray(pcd.colors)
#
## Print each point with its color and index
#for i in range(len(points)):
#    print(f'Point #{i}: {points[i]}, Color: {colors[i]}')
import open3d as o3d
import os
import numpy as np
from collections import Counter

print(os.getcwd())

print("specify name to open the pcd file")
path = input()
pcd = o3d.io.read_point_cloud(f"{path}.pcd")
o3d.visualization.draw_geometries([pcd])

colors = np.asarray(pcd.colors)

# Convert colors to string representation
colors_str = [str(color) for color in colors]

# Count the occurrence of each unique color
color_counts = Counter(colors_str)

# Print each unique color and its count
for color, count in color_counts.items():
    print(f'Color: {color}, Count: {count}')

# Print the number of unique colors
print(f'Number of unique colors: {len(color_counts)}')

