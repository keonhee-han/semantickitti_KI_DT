import numpy as np
import cv2
from PIL import Image
import torch

import os, glob


class SemanticKittiDataLoader:
    def __init__(self, data_dir, sequences, train=True):
        self.data_dir = data_dir
        self.sequences = sequences
        self.train = train
        self.cmap = create_semkitti_label_colormap()

    def load_data(self, sequence):
        """Loads and processes data for each sequence."""
        sequence_dir = os.path.join(self.data_dir, sequence)

        self.K_path = os.path.join(sequence_dir, "K_cam2.npy")
        self.E_path = os.path.join(sequence_dir, "T_cam2_velo.npy")

        # New paths for other data files
        self.dino_features_path = os.path.join(sequence_dir, "dino_features", f"{sequence}.npy")  # Assuming single file
        
        # Find all files using glob
        items = sorted(glob.glob(os.path.join(sequence_dir, "sam_segments", "*.png")))
        # Extract indices from filenames
        indices = [i[-10:-4] for i in items]

        # Load and process data for all files in each category
        dino_feats, pseudo_labelmaps = [], []
        
        # Load data
        K_ = np.load(self.K_path)
        E_ = np.load(self.E_path)
        self.H_, self.W_ = np.array(Image.open(items[0])).shape[:2]

        
        for idx in indices:     ## Iterating all files for dinofeatures, velodyne, velo_labels, sam_segments
            dino_feat = np.load(os.path.join(sequence_dir, "dino_features", f"{idx}.npy"))
            dino_feats.append(dino_feat)
            velo = np.fromfile(os.path.join(sequence_dir, "velodyne", f"{idx}.bin"), dtype=np.float32).reshape(-1, 4)
            velo[:, 3:] = 1  # Make homogeneous coordinates
            # velos.append(velo)
            velodyne_labels = np.fromfile(os.path.join(sequence_dir, "velodyne_labels", f"{idx}.label"), dtype=np.uint32)
            # velo_labels.append(velodyne_labels)
            img_sam = np.array(Image.open(os.path.join(sequence_dir, "sam_segments", f"{idx}.png")))
            # img_sams.append(img_sam)

            # Project velodyne points to image
            world2cam = E_ @ velo.T
            cam2img = K_ @ world2cam
            mask_invalid = cam2img[-1, :] > 0  # Discard points behind the front camera
            img2pixel = cam2img[:2, :] / cam2img[2, :]  # Normalize and get pixel coordinates
            img2pixel = np.concatenate([img2pixel, velodyne_labels[None, :]], axis=0)  # Add labels

            # Check projection validity and filter
            img2pixel = img2pixel.T.astype(np.int32)
            img2pixel = self.proj_val(img2pixel, mask_invalid)

            # Create labelmap, pseudo_labelmap, and colormap
            labelmap = np.zeros((self.H_, self.W_, 4), dtype=np.uint8)
            pseudo_labelmap = np.zeros((self.H_, self.W_, 4), dtype=np.uint8)   # Added 4th channel for SAM segments, last channel is to identify its label.
            labelmap_cmap = self.cmap[img2pixel[:, 2]]
            labelmap[..., :3][img2pixel[:, 1], img2pixel[:, 0]] = labelmap_cmap
            labelmap[..., -1][img2pixel[:, 1], img2pixel[:, 0]] = img2pixel[:, 2]
            # labelmap[img2pixel[:, 1], img2pixel[:, 0]] = labelmap_cmap

            # Process SAM segments and create pseudo_labelmap
            self.process_sam_segments(img_sam, labelmap, pseudo_labelmap, img2pixel[:, 2])
            pseudo_labelmaps.append(pseudo_labelmap)

        return dino_feats, pseudo_labelmaps,   # Return lists of processed data

    def proj_val(self, points, mask_invalid):
        """Filter valid projections."""
        points = points[mask_invalid, :]
        points = points[(points[:, 0] >= 0) & (points[:, 0] < self.W_)]
        points = points[(points[:, 1] >= 0) & (points[:, 1] < self.H_)]
        return points

    def process_sam_segments(self, img_sam, labelmap, pseudo_labelmap, velodyne_labels):
        """Process SAM segments and create pseudo_labelmap."""
        unique_sam_labels = np.unique(img_sam)
        for label in unique_sam_labels[1:]:  # Exclude background label
            mask = img_sam == label
            segment_cmap = labelmap[mask]
            unique_labels, counts = np.unique(segment_cmap, axis=0, return_counts=True)
            if len(unique_labels) == 1 and unique_labels[0].tolist() == [0, 0, 0, 0]:
                continue  # Skip background segments

            # Find dominant label (excluding background)
            max_counts = np.argmax(counts[1:])  # Remove background label
            dominant_clabel = unique_labels[max_counts + 1]

            # Assign dominant label to pseudo-labelmap for the segment
            pseudo_labelmap[mask] = dominant_clabel
            
            # return pseudo_labelmap
    
    def __getitem__(self, index):   ## Assuming index is the sequence number
        """Returns data and labels for a batch of images within a sequence."""
        
        dino_feat, pseudo_labelmap = self.load_data(self.sequences[index])  # Load data for the specified sequence

        return dino_feat, pseudo_labelmap  # Return the processed data and labels

    def __len__(self):
        # Return the number of items in the dataset
        # print(f"__Length of sequences: {len(self.sequences)} with train flag: {self.train}")
        return len(self.sequences)  # Assuming self.data is a list of your data items


def create_semkitti_label_colormap():
    """Creates a label colormap used in SEMANTICKITTI segmentation benchmark.
    Returns:
    A colormap for visualizing segmentation results in BGR format.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [245, 150, 100]
    # "car"
    colormap[2] = [245, 230, 100]
    # "bicycle"
    colormap[3] = [150, 60, 30]
    # "motorcycle"
    colormap[4] = [180, 30, 80]
    # "truck"
    colormap[5] = [255, 0, 0]
    # "other-vehicle"
    colormap[6] = [30, 30, 255]
    # "person"
    colormap[7] = [200, 40, 255]
    # "bicyclist"
    colormap[8] = [90, 30, 150]
    # "motorcyclist"
    colormap[9] = [255, 0, 255]
    # "road"
    colormap[10] = [255, 150, 255]
    # "parking"
    colormap[11] = [75, 0, 75]
    # "sidewalk"
    colormap[12] = [75, 0, 175]
    # "other-ground"
    colormap[13] = [0, 200, 255]
    # "building"
    colormap[14] = [50, 120, 255]
    # "fence"
    colormap[15] = [0, 175, 0]
    # "vegetation"
    colormap[16] = [0, 60, 135]
    # "trunk"
    colormap[17] = [80, 240, 150]
    # "terrain"
    colormap[18] = [150, 240, 255]
    # "pole"
    colormap[19] = [0, 0, 255]
    # "traffic-sign"
    return colormap