import numpy as np
import open3d as o3d

def color_to_label_old(pcd):
    colors = np.asarray(pcd.colors)
    unique_colors, labels = np.unique(colors, axis=0, return_inverse=True)
    return labels


'''
Overview:
This script reads a PCD file, extracts the colors, maps each unique color to a unique label, and then writes out the labels to a binary file.

Keep in mind that this is a simplified example and might not work directly with your data. You will likely need to adjust it based on the specifics of your data and the SemanticKITTI label file format.

Also, please note that this method will only work correctly if each unique color in your PCD files corresponds to a unique instance. If the same color is used for multiple instances or if multiple colors are used for the same instance, this method will not produce correct results.

File structure:
SemanticKITTI uses 32-bit integers to represent labels, and each point in the point cloud has a corresponding label in the label files. The labels are stored in binary format, so the files are not human-readable.

The label for each point represents the semantic class of that point. For example, a label of 1 might represent "car", a label of 2 might represent "pedestrian", and so on. The exact meaning of the labels is defined in a label map, which is a mapping from label values to semantic classes. SemanticKITTI provides a standard label map Link: https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml.
'''

def color_to_label(pcd, color_to_label_map):
    r'''dictionary where each key is a color (as represented in your PCD file) and each value is a label (as defined by the SemanticKITTI label map).

    '''
    colors = np.asarray(pcd.colors)
    # Convert colors to 8-bit RGB
    colors = (colors * 255).astype(np.uint8)
    labels = np.zeros(colors.shape[0], dtype=np.uint32)
    for idx, color in enumerate(colors):
        color_tuple = tuple(color)
        if color_tuple not in color_to_label_map:
            # Assign a new label to this color
            color_to_label_map[color_tuple] = len(color_to_label_map) + 1
        labels[idx] = color_to_label_map[color_tuple]
    return labels

def save_labels_old(labels, filename):
    np.savetxt(filename, labels, fmt='%i')

def save_labels(labels, filename):
    labels.astype(np.uint32).tofile(filename)

# Load a PCD file
print("__specify name to open the pcd file wo pcd format. e.g. 964merged_hpr")
path = input()
pcd = o3d.io.read_point_cloud(f"{path}.pcd")

# Initialize color-to-label map
color_to_label_map = {}

# Convert colors to labels
labels = color_to_label(pcd, color_to_label_map)

# Save the labels to a file
save_labels(labels, '{path}.label')
