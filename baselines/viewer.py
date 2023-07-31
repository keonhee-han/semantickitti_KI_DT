import open3d as o3d
import os

print(os.getcwd())

print("specify name to open the pcd file")
path = input()
pcd = o3d.io.read_point_cloud(f"{path}.pcd")
o3d.visualization.draw_geometries([pcd])
