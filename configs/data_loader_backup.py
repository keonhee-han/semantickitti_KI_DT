import numpy as np
import cv2
from PIL import Image

def proj_val(points: np.array, mask_invalid: np.array) -> np.array:
    """ Function to check the validity of the projection
    :param points: 2D points to be projected
    :param mask_invalid: mask_invalid to filter the points"""
    points = points[mask_invalid, :]                  ## discarding projections behind the front facing camera
    points = points[(points[:, 0] >= 0) & (points[:, 0] < W_)] ## discarding points outside the image
    points = points[(points[:, 1] >= 0) & (points[:, 1] < H_)]
    return points

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


## sample instance for sanity check : preprocess the data
velo = np.fromfile("00/velodyne/000000.bin", dtype=np.float32).reshape(-1, 4)
velo[:, 3:] = 1  # making homogeneous coordinates
velodyne_labels = np.fromfile("00/velodyne_labels/000000.label", dtype=np.uint32)
# velodyne_labels = np.fromfile("00/velodyne_labels/000000.label", dtype=np.uint32)
K_ = np.load("00/K_cam2.npy")
E_ = np.load("00/T_cam2_velo.npy")

# img = Image.open(path)  ## for gray: .convert("RGB")
# pixels = img.load()

path = "00/sam_segments/000000.png"
## import the img in opencv
# img_sam = cv2.imread(path)
# img_sam = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)  ##  we read the gray image in BGR format, so we convert it to RGB

img_sam = np.array(Image.open(path))
H_, W_ = img_sam.shape[:2]

## project the velodyne points to the image
world2cam = E_ @ velo.T
cam2img = K_ @ world2cam
mask_invalid = cam2img[-1,:] > 0                    ## discarding projections behind the front facing camera
# cam2img = cam2img[:, mask_invalid]
img2pixel = cam2img / cam2img[2, :]         ## normalizing the z axis in pixel coordinates with focal length.

img2pixel = img2pixel[:2, :]
img2pixel = np.concatenate([img2pixel, velodyne_labels[None,:]], axis=0)  ## adding the labels to the pixel coordinates
## Checking projection's validity
img2pixel = img2pixel.T                    
img2pixel = img2pixel.astype(np.int32)
img2pixel = proj_val(img2pixel, mask_invalid)               ### (3, n_pts) with having each point info [width, height, label]



## create the labelmap and pseudo_labelmap
labelmap = np.zeros((H_, W_, 3), dtype=np.uint8)            ### (H_, W_, 3) with each pixel having color [B, G, R]
pseudo_labelmap = np.zeros((H_, W_, 3), dtype=np.uint8)     ### (H_, W_, 3) with each pixel having color [B, G, R]
## now scatter the points accoring to the labels
cmap = create_semkitti_label_colormap()
labelmap_cmap = cmap[img2pixel[:, 2]]                       ### [n_pts, 3] with each point having color [B, G, R]
## scatter the points in the labelmap according to the pixel coordinates
labelmap[img2pixel[:, 1], img2pixel[:, 0]] = labelmap_cmap  ## scatter the points in the labelmap according to the pixel coordinates

unique_sam_labels = np.unique(img_sam)
## with unique same labels, we can iterate over all possible labels and get the pixel coordinates of each label. Code:
for label in unique_sam_labels[1:]: ## remove the background label
    mask = img_sam == label

    # # mask the labelmap, and count the possible labels within that masked SAM segment to assign the label into the segment.
    # # once you assign each of label of SAM segment, write them into the pseudo_labelmap
    # pixel_coords = np.argwhere(mask)        ## get the pixel coordinates of the mask
    # pixel_coords = pixel_coords[:, [1, 0]]  ### [width, height] format to match the labelmap format
    # # Now, get the labels of the pixel coordinates from the img2pixel
    # # get the labels of the pixel coordinates and count the unique labels and ignore the out of bounds:
    # mask_label = img2pixel[pixel_coords[:, 1], pixel_coords[:, 0], 2] != 0
    # labels = img2pixel[pixel_coords[:, 1], pixel_coords[:, 0]]
    # label = img2pixel[pixel_coords[:,0], pixel_coords[:,1]] 


    segment_cmap = labelmap[mask]    ## get the labels of the mask
    # mask_label = segment_cmap != [0,0,0] ## remove the background label
    # unique_labels, counts = np.unique(segment[mask_label[:,0]], return_counts=True)
    unique_labels, counts = np.unique(segment_cmap, axis=0, return_counts=True)
    if len(unique_labels) == 1 and unique_labels[0].tolist() == [0,0,0]:  ## remove the background label
        continue
    max_counts = np.argmax(counts[1:])  ## remove the background label
    dominant_clabel = unique_labels[max_counts + 1]
    pseudo_labelmap[mask] = dominant_clabel

    # pixel_coords = np.argwhere(mask)
    # # print("label", label, "pixel_coords", pixel_coords)
    # labelmap[pixel_coords[:,0], pixel_coords[:,1]]


## vizualize the image
cv2.imshow('image', pseudo_labelmap)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img = cv2.resize(img2pixel, (W_, H_), interpolation=cv2.INTER_LINEAR)  ## interpolated the pixel coordinates to the image size
# # how to resize with 
# img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
# img_norm = (img_norm * 255).astype(np.uint8)

# print("velo", velo.shape)
# print("velo_lab", velodyne_labels.shape)
# print("img", np.array(img).shape)
# print("shape", np.unique(np.array(img)))