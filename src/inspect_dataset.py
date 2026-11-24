
from pathlib import Path
from PIL import Image

# Dataset location
DATASET_PATH = Path("dataset/flower_photos")

print("=" * 45)
print("       FLOWER DATASET INSPECTION")
print("=" * 45)

if not DATASET_PATH.exists():
    raise FileNotFoundError("Dataset not found!")

total_images = 0

for category in sorted(DATASET_PATH.iterdir()):

    if not category.is_dir():
        continue

    image_files = list(category.glob("*"))
    valid_images = 0
    corrupted_images = 0

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                img.verify()
            valid_images += 1
        except Exception:
            corrupted_images += 1

    total_images += valid_images

    print(f"\nCategory: {category.name}")
    print(f"Valid images: {valid_images}")
    print(f"Corrupted images: {corrupted_images}")

print("\n" + "=" * 45)
print("TOTAL VALID IMAGES:", total_images)
print("=" * 45)
