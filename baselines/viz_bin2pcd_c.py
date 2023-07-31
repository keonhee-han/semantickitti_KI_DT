# Importing necessary libraries
import warnings

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# Suppress Numba warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

def visualize(gt_points, pred_points, gt_labels=None, pred_labels=None, method='PCA'):
    reducer = None
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 'LDA':
        reducer = LDA(n_components=2)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2)
    elif method == 'UMAP':
        reducer = umap.UMAP()
        
    if reducer:
        gt_points_reduced = reducer.fit_transform(gt_points)
        pred_points_reduced = reducer.fit_transform(pred_points)
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].scatter(gt_points_reduced[:, 0], gt_points_reduced[:, 1], c=gt_labels)
        axs[0].set_title('Ground Truth ' + method)
        axs[1].scatter(pred_points_reduced[:, 0], pred_points_reduced[:, 1], c=pred_labels)
        axs[1].set_title('Prediction ' + method)
        plt.show()

# Read the binary file | label file
print("specify: /path/to/your/pointcloud.bin")
path = input()
points = np.fromfile(f"{path}", dtype=np.float32).reshape(-1, 4)

labels = None
print("specify corresponding label file if existed. type z if none.")
path_label = input()
if path_label != "z":
    labels = np.fromfile(f"{path_label}", dtype=np.uint32)
    colormap = {
        0: [1, 0, 0],  # Red for label 0
        1: [0, 1, 0],  # Green for label 1
        2: [0, 0, 1],  # Blue for label 2
    }
    colors = []
    for label in labels:
        if label not in colormap:
            colormap[label] = np.random.rand(3).tolist()
        colors.append(colormap[label])
    colors = np.array(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

print("specify corresponding PCD file to compare with. e.g. /path/to/your/predicted.pcd type z if none.")
path_pcd = input()
if path_pcd != "z":
    predicted_pcd = o3d.io.read_point_cloud(path_pcd)
    points_pred = np.asarray(predicted_pcd.points)
    print("points_pred", points_pred.shape)
    colors_pred = np.asarray(predicted_pcd.colors)
    # colors_flattened_pred = colors_pred.flatten()
    # colors_flattened_pred = np.full(len(points_pred), 0) # Dummy color labels

    # Convert RGB colors to hex colors
    colors_flattened_pred = [mcolors.to_hex(c) for c in colors_pred]


    # Run dimensionality reduction methods on both ground truth and prediction
    visualize(points, points_pred, labels, colors_flattened_pred, method='PCA')
    #if labels is not None:
    #    visualize(points, points_pred, labels, colors_flattened_pred, method='LDA')
    visualize(points, points_pred, labels, colors_flattened_pred, method='t-SNE')
    visualize(points, points_pred, labels, colors_flattened_pred, method='UMAP')
