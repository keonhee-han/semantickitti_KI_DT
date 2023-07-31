'''
process:
    Read the PCD file: Use a library like open3d or pcl to read the PCD file.

    Create a color-to-label mapping: This would be a dictionary where each key is a color (as represented in your PCD file) and each value is a label (as defined by the SemanticKITTI label map).

    Convert colors to labels: For each point in the PCD file, look up its color in the color-to-label mapping and replace the color with the corresponding label.

    Write the labels to a file: Write the labels to a binary file, with one 32-bit integer per label.
'''
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

import warnings
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# Suppress Numba warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)


# Read the binary file | label file
print("specify: /path/to/your/pointcloud.bin")
path = input()
points = np.fromfile(f"{path}", dtype=np.float32).reshape(-1, 4)

print("specify correspodning label file if existed. type z if none.")
path_label = input()
if path_label != "z":
    labels = np.fromfile(f"{path_label}", dtype=np.uint32)
    
    # Create a colormap (replace this with your actual colormap)
    colormap = {
        0: [1, 0, 0],  # Red for label 0
        1: [0, 1, 0],  # Green for label 1
        2: [0, 0, 1],  # Blue for label 2
        # Add more colors as needed
    }
    
    # Assign each point a color based on its label
    colors = []
    for label in labels:
        if label not in colormap:
            # If label is not in colormap, assign a random color
            colormap[label] = np.random.rand(3).tolist()
        colors.append(colormap[label])
    colors = np.array(colors)

    # Assign each point a color based on its label
    #colors = np.array([colormap[label] for label in labels])
pcd = o3d.geometry.PointCloud()
# Create a point cloud from the points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # use only the first 3 columns (X, Y, Z)
pcd.colors = o3d.utility.Vector3dVector(colors)

print("specify correspodning PCD file to compared with. e.g. /path/to/your/predicted.pcd type z if none.")
path_pcd = input()
if path_pcd != "z":
    # Load the predicted point cloud from the PCD file
    predicted_pcd_path = path_pcd
    predicted_pcd = o3d.io.read_point_cloud(predicted_pcd_path)

    # Extract points and colors from the PCD file
    points = np.asarray(predicted_pcd.points)
    colors = np.asarray(predicted_pcd.colors)

    # Flatten colors to a single dimension
    colors_flattened = colors.flatten()

    # PCA
    pca = PCA(n_components=2)
    points_pca = pca.fit_transform(points)

    plt.figure()
    plt.scatter(points_pca[:, 0], points_pca[:, 1], c=colors_flattened)
    plt.title('PCA')
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2)
    points_tsne = tsne.fit_transform(points)

    plt.figure()
    plt.scatter(points_tsne[:, 0], points_tsne[:, 1], c=colors_flattened)
    plt.title('t-SNE')
    plt.show()


# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# Dimensionality Reduction

# PCA
pca = PCA(n_components=2)
points_pca = pca.fit_transform(points)

plt.figure()
plt.scatter(points_pca[:, 0], points_pca[:, 1], c=labels)
plt.title('PCA')
print("__visualizing PCA")
plt.show()

# LDA
lda = LDA(n_components=2)
points_lda = lda.fit_transform(points, labels)

plt.figure()
plt.scatter(points_lda[:, 0], points_lda[:, 1], c=labels)
plt.title('LDA')
print("__visualizing LDA")
plt.show()

print("__FYI, t-SNE requires heavy computation. This might takes longer time.")
# t-SNE
tsne = TSNE(n_components=2)
points_tsne = tsne.fit_transform(points)

plt.figure()
plt.scatter(points_tsne[:, 0], points_tsne[:, 1], c=labels)
plt.title('t-SNE')
print("__viz t-SNE")
plt.show()

# UMAP
reducer = umap.UMAP()
points_umap = reducer.fit_transform(points)

plt.figure()
plt.scatter(points_umap[:, 0], points_umap[:, 1], c=labels)
plt.title('UMAP')
print("__viz UMAP")
plt.show()
