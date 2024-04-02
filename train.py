"""
Defines a train function that takes the model, loss function, optimizer, configuration, and data loaders as input.
Uses the device variable to check for GPU availability and moves the model to the appropriate device.
Implements a training loop with the following steps for each epoch:
    Sets the model to training mode.
    Tracks the running loss during training.
    Iterates through the training data loader.
    Moves data and labels to the device.
    Performs a forward pass to get model predictions.
    Calculates the loss using the chosen criterion.
    Performs backpropagation and updates the optimizer.
    Updates the running loss.
    Prints the epoch loss after a full pass over the training data.
Optionally performs validation using a provided validation data loader:
    Sets the model to evaluation mode.
    Tracks the validation loss.
    Disables gradient calculation with torch.no_grad() for efficiency.
    Iterates through the validation data loader.
    Calculates the loss on each validation batch.
    Calculates the average validation loss.
    Prints the average validation loss after a full pass over the validation data.
Prints a completion message after all epochs.
"""

import torch
import cv2

def train(model, criterion, optimizer, config, train_loader, val_loader=None):
  """Train the semantic segmentation model."""

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  one_hot = False

  # Training loop
  for epoch in range(config['epochs']):
    print(f"\nEpoch: {epoch+1}/{config['epochs']}")

    # Train phase
    model.train()
    running_loss = 0.0
    for DiNO_feats, pseudo_labels in train_loader:
      inputs = [i.to(device) for i in DiNO_feats]
      labels_tr = [i.to(device) for i in pseudo_labels]
      outputs, losses = [], []

      # Forward pass
      for i in range(len(inputs)):
        # output = model(inputs[i].squeeze(0).permute(2, 0, 1)) # Permute input for channel-first format (NCHW)
        indices_map = labels_tr[i][..., -1].squeeze(0)
        output = model(inputs[i], tuple(indices_map.shape))
        if one_hot == True:
            target = one_hot_encode(indices_map).permute(2, 0, 1).unsqueeze(0)
        else:
            target = indices_map.unsqueeze(0)
        loss, acc_train, mIoU_train = criterion(output, target)
        # losses.append(loss)
        running_loss += loss.item()

    #   outputs = model(inputs)
    #   loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      # Update statistics
    #   running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"[Train] Loss: {epoch_loss:.4f}")
    print(f"[Train] accuracy: {acc_train:.4f}")
    print(f"[Train] mIoU: {mIoU_train:.4f}")

    # Validation phase
    if val_loader:
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
        for DiNO_feats, pseudo_labels in val_loader:
          # inputs, labels = inputs.to(device), labels.to(device)
          inputs = [i.to(device) for i in DiNO_feats]
          labels = [i.to(device) for i in pseudo_labels]
          outputs, losses = [], []

          # Forward pass
          for i in range(len(inputs)):
            indices_map = labels[i][..., -1].squeeze(0)
            output = model(inputs[i], tuple(indices_map.shape))
            if one_hot == True:
                target = one_hot_encode(indices_map).permute(2, 0, 1).unsqueeze(0)
            else:
                target = indices_map.unsqueeze(0)
            loss, acc_val, mIoU_val = criterion(output, target)
            val_loss += loss.item()
          # outputs = model(inputs)
          # val_loss += criterion(outputs, labels).item()
      val_loss /= len(val_loader)
      print(f"[Val] Loss: {val_loss:.4f}")
      print(f"[Val] accuracy: {acc_val:.4f}")
      print(f"[Val] mIoU: {mIoU_val:.4f}")

  print("__Training completed!")

  print("__Visualizing both training and testing...!")
  # Display all images in labels
  display_images(labels, 'Test_pseudo_labelmap')

  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # Display all images in labels_tr
  display_images(labels_tr, 'Training_pseudo_labelmap')

  cv2.waitKey(0)
  cv2.destroyAllWindows()

def display_images(images, title):
    # Convert all images to BGR format and concatenate them
    concatenated_image = cv2.vconcat([cv2.cvtColor(img[...,:3].squeeze(0).numpy(), cv2.COLOR_RGB2BGR) for img in images])

    # Display the concatenated image
    cv2.imshow(title, concatenated_image)


def one_hot_encode(labels, num_classes=19):
    # Create a tensor of zeros with the same size as the labels tensor, but with an extra dimension for the classes
    one_hot = torch.zeros(*labels.shape, num_classes, device=labels.device)

    # Use scatter_ to write ones into the tensor at the indices specified by the labels
    one_hot.scatter_(-1, labels.unsqueeze(-1).long(), 1)

    return one_hot