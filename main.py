import torch
from torch.utils.data import DataLoader

from configs.data_loader import SemanticKittiDataLoader  # Assuming data loader class
from models.semantic_segmentation_mlp import SemanticSegmentationModel
from configs.loss import CrossEntropyLoss  # Assuming CrossEntropyLoss class
from configs.optimizer import AdamOptimizer  # Assuming AdamOptimizer class
from train import train  # Assuming train function resides in train.py

import json

def load_config(config_path):
  """Loads training configuration from a JSON file."""
  with open(config_path, 'r') as f:
    config = json.load(f)
  return config


def main():
  """Entry point for training the semantic segmentation model."""

  # Load configuration
  config = load_config("configs/config.json")  # Assuming config.json exists in the same directory

  # Prepare data loaders
  train_sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]  # Specify training sequences
  train_loader = DataLoader(
      SemanticKittiDataLoader(config["data_dir"], sequences=train_sequences, train=True),
      batch_size=config["batch_size"],
      shuffle=True,
      num_workers=config["num_workers"],
      pin_memory=True,
  )

  test_sequence = "08"  # testing sequence
  test_loader = DataLoader(
      SemanticKittiDataLoader(config["data_dir"], sequences=[test_sequence], train=False),
      batch_size=config["batch_size"],
      shuffle=False,  # Don't shuffle for testing
      num_workers=config["num_workers"],
      pin_memory=True,
  )

  # Define model, loss function and optimizer
  model = SemanticSegmentationModel(config["model"]["in_features"], config["model"]["num_classes"])
  criterion = CrossEntropyLoss()
  optimizer = AdamOptimizer(model.parameters(), lr=config["optimizer"]["learning_rate"])

  # Train the model
  # Note: Using test_loader for validation here, as it's specifically for sequence 08
  train(model, criterion, optimizer, config, train_loader, val_loader=test_loader)


if __name__ == "__main__":
  main()