
import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = Path("dataset/flower_photos")
MODEL_PATH = Path("models/flower_classifier.pth")
METADATA_PATH = Path("dataset/dataset_metadata.json")

IMAGE_SIZE = 224
BATCH_SIZE = 32
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("       FLOWER IMAGE RECOGNITION TRAINING")
print("=" * 50)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ==========================================
# 2. LOAD METADATA
# ==========================================

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

class_names = metadata["classes"]
num_classes = len(class_names)

print("Categories:", class_names)

# ==========================================
# 3. IMAGE TRANSFORMATIONS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================================
# 4. RECREATE DATASET SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

base_dataset = datasets.ImageFolder(DATASET_PATH)
targets = base_dataset.targets

indices = list(range(len(base_dataset)))

train_indices, remaining_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=targets
)

val_indices, _ = train_test_split(
    remaining_indices,
    test_size=0.50,
    random_state=42,
    stratify=[targets[i] for i in remaining_indices]
)

train_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

eval_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=eval_transform
)

train_set = torch.utils.data.Subset(
    train_dataset, train_indices
)

val_set = torch.utils.data.Subset(
    eval_dataset, val_indices
)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# ==========================================
# 5. LOAD PRETRAINED MOBILENETV2
# ==========================================

print("\nLoading pretrained MobileNetV2...")

weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(weights=weights)

# Freeze feature extractor initially
for param in model.features.parameters():
    param.requires_grad = False

# Replace original 1000-class output
model.classifier[1] = nn.Linear(
    model.last_channel,
    num_classes
)

model = model.to(device)

print("Model loaded successfully!")
print("Output categories:", num_classes)

# ==========================================
# 6. LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

best_accuracy = 0.0
best_state = None

# ==========================================
# 7. TRAINING FUNCTION
# ==========================================

def train_model(epochs, optimizer, phase):

    global best_accuracy, best_state

    for epoch in range(epochs):

        model.train()

        # Keep frozen backbone in evaluation mode
        if phase == "Warmup":
            model.features.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"{phase} Epoch {epoch + 1}/{epochs}"
        )

        for images, labels in progress:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = running_loss / total
        train_accuracy = correct / total

        # ==================================
        # VALIDATION
        # ==================================

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        print(f"\n{phase} - Epoch {epoch + 1}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Save best model based on validation accuracy
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save({
                "model_state_dict": best_state,
                "classes": class_names,
                "image_size": IMAGE_SIZE,
                "architecture": "MobileNetV2",
                "best_validation_accuracy": best_accuracy
            }, MODEL_PATH)

            print("Best model checkpoint saved!")

# ==========================================
# 8. PHASE 1: CLASSIFIER TRAINING
# ==========================================

print("\n========== PHASE 1: WARMUP ==========")

optimizer = optim.AdamW(
    model.classifier.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

train_model(
    WARMUP_EPOCHS,
    optimizer,
    "Warmup"
)

# ==========================================
# 9. PHASE 2: FINE-TUNING
# ==========================================

print("\n========== PHASE 2: FINE-TUNING ==========")

# Unfreeze final three feature blocks
for block in list(model.features.children())[-3:]:
    for param in block.parameters():
        param.requires_grad = True

optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.0001,
    weight_decay=0.0001
)

train_model(
    FINETUNE_EPOCHS,
    optimizer,
    "Fine-tuning"
)

# ==========================================
# 10. RESTORE BEST CHECKPOINT
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("\n" + "=" * 50)
print("       TRAINING COMPLETED")
print("=" * 50)

print("Best validation accuracy:",
      f"{best_accuracy * 100:.2f}%")

print("Saved model:", MODEL_PATH)
print("Device used:", device)
