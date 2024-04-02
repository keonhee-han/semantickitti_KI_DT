import torch
from torch import nn
from torch.nn import functional as F

class SemanticSegmentationModel(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=128, num_mlp_blocks=1):
        super(SemanticSegmentationModel, self).__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_mlp_blocks = num_mlp_blocks

        # Input projection layer
        self.project_in = nn.Linear(in_features, hidden_dim)

        # MLP processing blocks
        self.mlp_blocks = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_dim * 4),  # Expansion layer
            nn.LeakyReLU(),  # Activation function
            nn.Linear(hidden_dim * 4, hidden_dim),  # Reduction layer
            # nn.Dropout(p=0.2)  # Dropout for regularization (adjust probability)
        ] * num_mlp_blocks)

        # Output layer for segmentation prediction
        self.final = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, resol=(375, 1242)):    # Added resol parameter which varies depending on image inputs
        """Processes input features and generates segmentation prediction."""
        x = x.float()  # Convert x to a Float tensor
        batch_size, h, w, in_features = x.size()

        # Reshape input for efficient processing
        x = x.view(batch_size, h * w, in_features)

        if x.shape[0] == 1: x = x.squeeze(0)

        # Project features to a higher dimension
        x = self.project_in(x)

        # Process features through multiple MLP blocks
        for _ in range(self.num_mlp_blocks):
            x = self.mlp_blocks(x)
            
        # Final prediction layer
        pred = self.final(x)

        # Reshape back to image-like format for final prediction
        pred = pred.view(batch_size, self.num_classes, h, w)

        # Upsample prediction to match original image size
        # pred = nn.functional.interpolate(pred, size=(375, 1242), mode="bilinear")
        pred = nn.functional.interpolate(pred, size=resol, mode="bilinear")

        # Apply softmax to the class dimension
        pred = F.softmax(pred, dim=1)

        return pred