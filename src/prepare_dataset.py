
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# 1. Configuration
# ------------------------------------------------

DATASET_PATH = Path("dataset/flower_photos")
OUTPUT_PATH = Path("dataset")

BATCH_SIZE = 32
IMAGE_SIZE = 224
SEED = 42

torch.manual_seed(SEED)

# ------------------------------------------------
# 2. Image preprocessing
# ------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------
# 3. Load dataset
# ------------------------------------------------

base_dataset = datasets.ImageFolder(DATASET_PATH)

class_names = base_dataset.classes
targets = base_dataset.targets

print("\nFlower categories:", class_names)
print("Total images:", len(base_dataset))

# ------------------------------------------------
# 4. Stratified dataset splitting
# ------------------------------------------------

all_indices = list(range(len(base_dataset)))

train_indices, remaining_indices = train_test_split(
    all_indices,
    test_size=0.30,
    random_state=SEED,
    stratify=targets
)

val_indices, test_indices = train_test_split(
    remaining_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=[targets[i] for i in remaining_indices]
)

# ------------------------------------------------
# 5. Apply separate transformations
# ------------------------------------------------

train_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

eval_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=eval_transform
)

train_set = Subset(train_dataset, train_indices)
val_set = Subset(eval_dataset, val_indices)
test_set = Subset(eval_dataset, test_indices)

# ------------------------------------------------
# 6. Create DataLoaders
# ------------------------------------------------

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ------------------------------------------------
# 7. Save dataset information
# ------------------------------------------------

metadata = {
    "classes": class_names,
    "class_to_idx": base_dataset.class_to_idx,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "train_count": len(train_set),
    "validation_count": len(val_set),
    "test_count": len(test_set)
}

OUTPUT_PATH.joinpath("dataset_metadata.json").write_text(
    json.dumps(metadata, indent=4)
)

# ------------------------------------------------
# 8. Display results
# ------------------------------------------------

print("\n========== DATASET SPLIT ==========")
print("Training images  :", len(train_set))
print("Validation images:", len(val_set))
print("Testing images   :", len(test_set))

print("\n========== DATALOADERS ==========")
print("Training batches :", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches  :", len(test_loader))

# ------------------------------------------------
# 9. Verify image batch
# ------------------------------------------------

images, labels = next(iter(train_loader))

print("\n========== IMAGE VERIFICATION ==========")
print("Image tensor shape:", images.shape)
print("Label tensor shape:", labels.shape)
print("Image data type:", images.dtype)

print("\nDataset preparation completed successfully!")
