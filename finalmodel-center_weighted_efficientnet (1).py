import os
import sys
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.models import mobilenet_v2
from efficientnet_pytorch import EfficientNet

from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Custom dataset class to load images directly from catalog paths
class GalaxyDataset(Dataset):
    def __init__(self, catalog_path, images_dir, transform=None):
        self.catalog = pd.read_parquet(catalog_path)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        row = self.catalog.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filename'])  # Adjust column name if needed
        image = Image.open(img_path).convert("RGB")
        label = row['label']  # Assuming 'label' column contains the class labels

        if self.transform:
            image = self.transform(image)
        return image, label

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        """
        Focal Loss to address class imbalance by focusing on hard examples.
        Parameters:
            alpha (Tensor or None): Weighting factor for each class. Shape: [num_classes].
            gamma (float): Focusing parameter. Default is 2.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for class imbalance
        self.gamma = gamma  # Focusing parameter

    def forward(self, inputs, targets):
        """
        Forward pass of Focal Loss.
        Args:
            inputs: Predicted logits from the model (before softmax). Shape: [batch_size, num_classes].
            targets: Ground-truth class labels. Shape: [batch_size].
        Returns:
            Focal Loss value.
        """
        # Compute Cross-Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # Shape: [batch_size]
        # Convert logits to probabilities
        pt = torch.exp(-ce_loss)  # Probability of the correct class. Shape: [batch_size]
        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # Get the alpha weight for each target in the batch
            focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        else:
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()  # Average over the batch

# Define data transformations
image_size = 224
train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    #transforms.RandomAffine(degrees=90, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(degrees=20), ### added augmentation
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),### added augmentation
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to image catalogs and images directory
images_dir = 'zoobot_data/galaxy_mnist/images'
train_catalog_path = 'zoobot_data/galaxy_mnist/galaxy_mnist_train_catalog.parquet'
test_catalog_path = 'zoobot_data/galaxy_mnist/galaxy_mnist_test_catalog.parquet'

# Create datasets
train_dataset = GalaxyDataset(train_catalog_path, images_dir, transform=train_transforms)
val_dataset = GalaxyDataset(test_catalog_path, images_dir, transform=val_transforms)

batch_size = 32

# Recreate data loaders with the optimal batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class CenterWeightedEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CenterWeightedEfficientNet, self).__init__()
        # Load EfficientNet base
        self.model = EfficientNet.from_name('efficientnet-b0')
        # Remove the existing global average pooling layer
        self.feature_extractor = self.model.extract_features
        # Replace the classifier with a new fully connected layer
        self.fc = nn.Linear(self.model._fc.in_features, num_classes)

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        b, c, h, w = features.size()  # Batch size, channels, height, width

        # Create a Gaussian center mask
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=features.device),
                              torch.linspace(-1, 1, w, device=features.device),
                              indexing="ij")  # Ensure tensors are on the same device
        distances = torch.sqrt(x ** 2 + y ** 2)
        center_mask = torch.exp(-((distances / 0.5) ** 2))  # Adjust sigma=0.5 for sharper center focus
        center_mask = center_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        center_mask = center_mask.to(features.device)  # Ensure it is on the same device as features

        # Apply the mask to the features
        weighted_features = features * center_mask
        # Pool the weighted features
        pooled = weighted_features.mean([2, 3])  # Average over height and width

        # Classify
        logits = self.fc(pooled)
        return logits


class GalaxyResNet18(nn.Module):
    def __init__(self, num_classes):
        super(GalaxyResNet18, self).__init__()
        self.model = resnet18(pretrained=False)  # Set pretrained=True if desired
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class GalaxyMobileNet(nn.Module):
    def __init__(self, num_classes):
        super(GalaxyMobileNet, self).__init__()
        self.model = mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)


# Define ViT model hyperparameters
patch_size = 28
dim = 256
depth = 2
heads = 4
mlp_dim = 1024
num_classes = 4
dropout = 0.01
emb_dropout = 0.1
run_note = "center weighted efficientnet-b0 - final re-run with GPU"

model_type = "center_weighted_efficientnet"

# if model_type == "cnn":
#     model = GalaxyCNN_binu(num_classes=num_classes).to(device)
# elif model_type == "vit":
#     model = GalaxyViT3(
#         image_size=image_size,
#         patch_size=patch_size,
#         num_classes=num_classes,
#         dim=dim,
#         depth=depth,
#         heads=heads,
#         mlp_dim=mlp_dim,
#         channels=3,
#         dropout=dropout,
#         emb_dropout=emb_dropout
#     ).to(device)
# elif model_type == "efficientnet":
#     model = GalaxyEfficientNet(num_classes=num_classes).to(device)
if model_type == "center_weighted_efficientnet":
    model = CenterWeightedEfficientNet(num_classes=num_classes).to(device)
elif model_type == "resnet18":
    model = GalaxyResNet18(num_classes=num_classes).to(device)
elif model_type == "mobilenet":
    model = GalaxyMobileNet(num_classes=num_classes).to(device)

else:
    raise ValueError("Invalid model type.")

# Training setup
criterion = nn.CrossEntropyLoss()  # if not using FocalLoss

# class_weights = [3.49, 6, 6, 3.70]  # For classes 0, 1, 2, 3
# alpha = torch.tensor(class_weights).to(device)
# criterion = FocalLoss(alpha=alpha, gamma=2)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
epochs = 20

# Track history
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

# Training function with progress bar and estimated time remaining
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10):
    start_time = time.time()
    average_epoch_time = 0  # Track average epoch time for estimated completion time

    # Track the final training and validation metrics
    final_train_loss, final_train_acc, final_val_loss, final_val_acc = 0, 0, 0, 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, unit="batch", leave=True, file=sys.stdout) as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                # Ensure tensors are in float32 for MPS compatibility
                images = images.to(torch.float32)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tepoch.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        # Update average epoch time
        epoch_time = time.time() - epoch_start_time
        average_epoch_time = ((average_epoch_time * epoch) + epoch_time) / (epoch + 1)

        # Estimate completion time
        remaining_time = average_epoch_time * (epochs - epoch - 1)
        finish_time = datetime.datetime.fromtimestamp(time.time() + remaining_time).strftime('%I:%M %p')

        # Save training metrics to history
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print(f"Estimated completion time: {finish_time}")

        # Validation phase
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion)

        # Save validation metrics to history
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

        scheduler.step()

        # Update final metrics after last epoch
        if epoch == epochs - 1:
            final_train_loss, final_train_acc = epoch_loss, epoch_acc
            final_val_loss, final_val_acc = val_loss, val_acc

    total_time = time.time() - start_time
    print(f"Total Training Time: {int(total_time // 60)} minutes, {int(total_time % 60)} seconds")

    return final_train_loss, final_train_acc, final_val_loss, final_val_acc, total_time


# Evaluation function that returns predictions and labels for confusion matrix
def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return loss, accuracy, all_preds, all_labels

# Function to plot accuracy, loss curves, and confusion matrix
def save_model(model, path, val_accuracy, model_type, hyperparameters, history, true_labels, pred_labels, class_names,
               final_train_loss, final_train_acc, final_val_loss, final_val_acc, total_time):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{path}/{model_type}_{timestamp}_val_acc{val_accuracy:.2f}"
    os.makedirs(folder_name, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), f"{folder_name}/GViT3_{timestamp}.pt")

    # Calculate total training time in minutes and seconds
    minutes, seconds = divmod(total_time, 60)

    # Format hyperparameters text
    hyperparams_text = (
        f"Timestamp: {timestamp}\n"
        f"Training Loss: {final_train_loss:.4f}, Training Accuracy: {final_train_acc:.2f}% ({len(train_loader.dataset)} images)\n"
        f"Validation Loss: {final_val_loss:.4f}, Validation Accuracy: {final_val_acc:.2f}% ({len(val_loader.dataset)} images)\n"
        f"Total Training Time: {int(minutes)} minutes, {int(seconds)} seconds\n\n"
        + "\n".join([f"{k}: {v}" for k, v in hyperparameters.items()])
    )

    # Write hyperparameters to text file
    hyperparams_path = f"{folder_name}/hyperparameters.txt"
    with open(hyperparams_path, 'w') as f:
        f.write(hyperparams_text)

    # Plot accuracy, loss curves, and confusion matrix in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Model Summary - {timestamp}")

    # Top-left: Hyperparameters text
    axes[0, 0].text(0.5, 0.5, hyperparams_text, ha='center', va='center', fontsize=10, wrap=True)
    axes[0, 0].set_title("Hyperparameters")
    axes[0, 0].axis('off')

    # Top-right: Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=axes[0, 1], cmap=plt.cm.Blues)
    axes[0, 1].set_title("Confusion Matrix")

    # Bottom-left: Accuracy plot
    epochs_range = range(1, len(history['accuracy']) + 1)  # Adjust range to start from 1
    axes[1, 0].plot(epochs_range, history['accuracy'], label='Train Accuracy')
    axes[1, 0].plot(epochs_range, history['val_accuracy'], label='Val Accuracy')
    axes[1, 0].set_title("Training and Validation Accuracy")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1, 0].legend()

    # Bottom-right: Loss plot
    axes[1, 1].plot(epochs_range, history['loss'], label='Train Loss')
    axes[1, 1].plot(epochs_range, history['val_loss'], label='Val Loss')
    axes[1, 1].set_title("Training and Validation Loss")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to include the main title
    plot_path = f"{folder_name}/acc_loss_cm.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved model, hyperparameters, and plots to '{folder_name}'")

# Run the training
final_train_loss, final_train_acc, final_val_loss, final_val_acc, total_time = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epochs
)

# Final evaluation
val_loss, val_acc, val_preds, val_labels = evaluate_model(model, val_loader, criterion)

# Hyperparameters
hyperparameters = {
    #"Patch Size": patch_size,
    #"Dimension": dim,
    #"Depth": depth,
    #"Heads": heads,
    #"MLP Dimension": mlp_dim,
    "Classes": num_classes,
    #"Dropout": dropout,
    #"Embedding Dropout": emb_dropout,
    "Learning Rate": optimizer.param_groups[0]["lr"],
    "Batch Size": batch_size,
    "Epochs": epochs,
    "Notes": run_note,
}

# Save model, hyperparameters, and plots
class_names = ["Round", "Cigar", "Edge", "Spiral"]
model_name = model_type
save_model(
    model, f"final_models/{model_name}", val_acc, model_type, hyperparameters, history,
    val_labels, val_preds, class_names, final_train_loss, final_train_acc,
    final_val_loss, final_val_acc, total_time
)