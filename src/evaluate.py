
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------
# 1. Configuration
# ------------------------------------------------

DATASET_PATH = Path("dataset/flower_photos")
MODEL_PATH = Path("models/flower_classifier.pth")
OUTPUT_DIR = Path("models")

BATCH_SIZE = 32
IMAGE_SIZE = 224
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Evaluation device:", device)

# ------------------------------------------------
# 2. Load checkpoint and metadata
# ------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

metadata = json.loads(
    Path("dataset/dataset_metadata.json").read_text()
)

class_names = checkpoint["classes"]
num_classes = len(class_names)

print("Architecture:", checkpoint["architecture"])
print("Flower categories:", class_names)

# ------------------------------------------------
# 3. Recreate evaluation preprocessing
# ------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------
# 4. Load dataset and reproduce exact split
# ------------------------------------------------

base_dataset = datasets.ImageFolder(DATASET_PATH)

if base_dataset.classes != class_names:
    raise ValueError("Dataset class order does not match model classes.")

all_indices = list(range(len(base_dataset)))
targets = base_dataset.targets

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

eval_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=eval_transform
)

test_set = Subset(eval_dataset, test_indices)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTest images:", len(test_set))

# ------------------------------------------------
# 5. Rebuild MobileNetV2 architecture
# ------------------------------------------------

model = models.mobilenet_v2(weights=None)

model.classifier[1] = torch.nn.Linear(
    model.last_channel,
    num_classes
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ------------------------------------------------
# 6. Run predictions
# ------------------------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

# ------------------------------------------------
# 7. Calculate evaluation metrics
# ------------------------------------------------

accuracy = accuracy_score(all_labels, all_predictions)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
)

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
)

print("\n========== TEST RESULTS ==========")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

print("\n========== CONFUSION MATRIX ==========")
print(matrix)

# ------------------------------------------------
# 8. Save evaluation report
# ------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report_path = OUTPUT_DIR / "evaluation_report.txt"

report_path.write_text(
    f"Architecture: MobileNetV2\n"
    f"Test images: {len(test_set)}\n"
    f"Test accuracy: {accuracy * 100:.2f}%\n\n"
    f"{report}\n"
    f"Confusion Matrix:\n{matrix}\n"
)

# ------------------------------------------------
# 9. Visualize confusion matrix
# ------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(matrix, interpolation="nearest", cmap="Blues")
plt.title("MobileNetV2 - Confusion Matrix")
plt.colorbar()

positions = np.arange(num_classes)

plt.xticks(positions, class_names, rotation=30)
plt.yticks(positions, class_names)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for row in range(num_classes):
    for col in range(num_classes):
        plt.text(
            col,
            row,
            str(matrix[row, col]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nEvaluation completed successfully!")
print("Report saved:", report_path)
print("Confusion matrix saved:", OUTPUT_DIR / "confusion_matrix.png")
