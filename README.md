# semantickitti_KI_DT
KI data tooling project (https://www.ki-datatooling.de/) under visual computing &amp; AI lab @ TUM

brief description with PPT file: https://docs.google.com/presentation/d/1XyEqDIZ3atFjfOsngh5KmbcIn4wPWwGMD5uNWHcleV4/edit?usp=sharing
![alt text](https://github.com/Kvasir8/semantickitti_KI_DT/blob/main/tSNE_GTvsPred.png)
![alt text](https://github.com/Kvasir8/semantickitti_KI_DT/blob/main/sample_bin2color_label.png)
# To visualize semantickitti point cloud and its corresponding label as color
Sample data for the test is available under test folder.
To visualize a .bin file in open3d, execute the following as an example.
```bash
cd baselines
python3 viz_bin2pcd_c.py
```
It will prompt to specify the .bin file and optionally the corresponding label file. Example to type:
```bash
test/000964.bin
test/000964.label
```
# To visualize the .pcd file
```bash
python3 viewer.py
test/964merged_hpr
```

it will show color labeled point cloud in a frame, which can be used for better intuition of how to find a corresponding color label for the prediction of .pcd file (where it doesn't contain class label information e.g. blue=ground, green=forest)

# Idea to annotate label from .pcd format with its ground truth label
Convert .bin files from SemanticKITTI to .pcd format.
Visualize both ground truth and predicted point clouds (which are color-annotated) using Open3D.
By comparing the colors, determine a mapping from colors to semantic labels.
Use this color-to-label mapping to convert the color-annotated point clouds to label-annotated point clouds.
Evaluate the predicted labels against the ground truth labels.
This is a reasonable approach, assuming there is a consistent correlation between colors and semantic labels in your predicted data and the SemanticKITTI ground truth.

Applying t-SNE or UMAP to visualize the color space may help understand the correlation between colors and semantic labels, especially if there are many unique colors. These dimensionality reduction techniques can help visualize high-dimensional data in 2D or 3D, making it easier to observe clusters of similar colors. If clusters correspond to semantic classes, this may help in creating a color-to-label mapping.

However, it's important to note that t-SNE and UMAP are stochastic methods and can produce different results on different runs. Also, they do not inherently provide a way to map from the high-dimensional space to the reduced space. The reduced space is intended primarily for visualization rather than for creating mappings. So, while these methods can help visualize the color space, you will likely still need to use other methods to create precise color-to-label mapping.
Finally, remember that this approach will only work if there is a clear and consistent correlation between colors and semantic labels in your data. If the same color is used for multiple semantic classes, or if multiple colors are used for the same semantic class, this approach may not work correctly.
