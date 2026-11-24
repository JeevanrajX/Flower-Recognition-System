from pathlib import Path
import uuid

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# -----------------------------------------
# Configuration
# -----------------------------------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flower_classifier.pth"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# This is a simple confidence-based rejection rule.
# It is a practical first safeguard, not a guaranteed
# detector of every image that is not a flower.
CONFIDENCE_THRESHOLD = 85.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------
# Load trained model
# -----------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Make sure models/flower_classifier.pth exists."
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

class_names = checkpoint["classes"]
image_size = checkpoint["image_size"]

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# -----------------------------------------
# Image preprocessing
# -----------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------------
# Helper functions
# -----------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_flower(image):
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = probabilities.argmax().item()
    confidence = probabilities[predicted_index].item() * 100

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "is_flower": False,
            "message": (
                "This image could not be confidently identified as one "
                "of the supported flower categories. Please upload a clear "
                "image of a daisy, dandelion, rose, sunflower, or tulip."
            ),
            "confidence": round(confidence, 2)
        }

    return {
        "is_flower": True,
        "flower": class_names[predicted_index],
        "confidence": round(confidence, 2),
        "probabilities": {
            name: round(probabilities[i].item() * 100, 2)
            for i, name in enumerate(class_names)
        }
    }


# -----------------------------------------
# Main page
# -----------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "Please select an image."

        else:
            file = request.files["image"]

            if file.filename == "":
                error = "No image selected."

            elif not allowed_file(file.filename):
                error = "Please upload JPG, JPEG, PNG or WEBP."

            else:
                try:
                    image = Image.open(file.stream).convert("RGB")
                    result = predict_flower(image)

                    extension = file.filename.rsplit(".", 1)[1].lower()
                    filename = secure_filename(f"{uuid.uuid4().hex}.{extension}")
                    image_path = UPLOAD_FOLDER / filename

                    image.save(image_path)
                    image_url = f"/static/uploads/{filename}"

                except (UnidentifiedImageError, OSError, ValueError):
                    error = "The uploaded file is not a valid image."

                except Exception as exc:
                    app.logger.exception("Image processing failed: %s", exc)
                    error = "Unable to process this image. Please try another."

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        error=error
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        result=None,
        image_url=None,
        error="Image is too large. Please upload an image smaller than 10 MB."
    ), 413


# -----------------------------------------
# Run application
# -----------------------------------------

if __name__ == "__main__":
    print("Flower Recognition System")
    print("Device:", device)
    print("Classes:", class_names)
    print("Rejection threshold:", f"{CONFIDENCE_THRESHOLD}%")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
