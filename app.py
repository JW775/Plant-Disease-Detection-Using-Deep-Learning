"""
Crop disease detection backend.

This backend uses Flask for APIs and a torchvision ResNet18 model for
45-class inference: 44 crop disease/healthy classes plus one background class.
"""
import io
import json
import logging
import os

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from typing import Optional, Union

# ==================== Configuration ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_INDICES_PATH = r"D:\tmp\plant_disease_backend\class_indices.json"
MODEL_PATH = os.path.join(BASE_DIR, "best_model_process.pth")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("Using device: %s", DEVICE)


# ==================== Class And Crop Metadata ====================
with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
    CLASS_INDICES = json.load(f)

CLASS_NAMES = [
    class_name
    for _, class_name in sorted(CLASS_INDICES.items(), key=lambda x: int(x[0]))
]
BG_CLASS_NAME = "bg"
logger.info("Loaded %s classes.", len(CLASS_NAMES))

CROP_CONFIG = {
    "Apple": {"icon": "🍎", "prefixes": ("Apple_",)},
    "Cherry": {"icon": "🍒", "prefixes": ("Cherry_",)},
    "Corn": {"icon": "🌽", "prefixes": ("Corn_(maize)_",)},
    "Grape": {"icon": "🍇", "prefixes": ("Grape_",)},
    "Orange": {"icon": "🍊", "prefixes": ("Orange_",)},
    "Peach": {"icon": "🍑", "prefixes": ("Peach_",)},
    "Bell Pepper": {"icon": "🌶️", "prefixes": ("Pepper,_bell_",)},
    "Potato": {"icon": "🥔", "prefixes": ("Potato_",)},
    "Rice": {"icon": "🌾", "prefixes": ("Rice_",)},
    "Squash": {"icon": "🎃", "prefixes": ("Squash_",)},
    "Strawberry": {"icon": "🍓", "prefixes": ("Strawberry_",)},
    "Tomato": {"icon": "🍅", "prefixes": ("Tomato_",)},
    "Wheat": {"icon": "🌾", "prefixes": ("Wheat_",)},
}

CROP_ALIASES = {
    "apple": "Apple",
    "苹果": "Apple",
    "cherry": "Cherry",
    "樱桃": "Cherry",
    "corn": "Corn",
    "maize": "Corn",
    "玉米": "Corn",
    "grape": "Grape",
    "葡萄": "Grape",
    "orange": "Orange",
    "citrus": "Orange",
    "柑橘": "Orange",
    "peach": "Peach",
    "桃": "Peach",
    "pepper": "Bell Pepper",
    "bell pepper": "Bell Pepper",
    "辣椒": "Bell Pepper",
    "potato": "Potato",
    "土豆": "Potato",
    "马铃薯": "Potato",
    "rice": "Rice",
    "水稻": "Rice",
    "squash": "Squash",
    "南瓜": "Squash",
    "strawberry": "Strawberry",
    "草莓": "Strawberry",
    "tomato": "Tomato",
    "西红柿": "Tomato",
    "番茄": "Tomato",
    "wheat": "Wheat",
    "小麦": "Wheat",
}

HEALTHY_TREATMENT = [
    "The plant appears healthy. Continue regular field monitoring.",
    "Keep irrigation and fertilization balanced.",
    "Check leaves regularly so early disease symptoms can be found quickly.",
]

DEFAULT_TREATMENT = [
    "Remove and isolate visibly infected leaves or plants.",
    "Improve ventilation and avoid excessive humidity.",
    "Use suitable fungicide or pesticide guidance from a local agricultural expert.",
    "Monitor disease development and treat early if symptoms spread.",
]


def normalize_crop_name(crop_name: str) -> str:
    crop_name = (crop_name or "").strip()
    return CROP_ALIASES.get(crop_name.lower(), CROP_ALIASES.get(crop_name, crop_name))


def class_to_crop(class_name: str) -> str | None:
    if class_name == BG_CLASS_NAME:
        return None
    for crop, config in CROP_CONFIG.items():
        if any(class_name.startswith(prefix) for prefix in config["prefixes"]):
            return crop
    return None


def get_crop_class_ids(crop_name: str) -> list[int]:
    crop_name = normalize_crop_name(crop_name)
    config = CROP_CONFIG.get(crop_name)
    if not config:
        return []
    return [
        idx
        for idx, class_name in enumerate(CLASS_NAMES)
        if any(class_name.startswith(prefix) for prefix in config["prefixes"])
    ]


def display_class_name(class_name: str) -> str:
    if class_name == BG_CLASS_NAME:
        return "Invalid non-crop image"

    name = class_name
    replacements = {
        "Apple_": "Apple ",
        "Cherry_(including_sour)_": "Cherry ",
        "Corn_(maize)_": "Corn ",
        "Grape_": "Grape ",
        "Orange_": "Orange ",
        "Peach_": "Peach ",
        "Pepper,_bell_": "Bell Pepper ",
        "Potato_": "Potato ",
        "Rice_": "Rice ",
        "Squash_": "Squash ",
        "Strawberry_": "Strawberry ",
        "Tomato_": "Tomato ",
        "Wheat_": "Wheat ",
    }
    for prefix, label in replacements.items():
        if name.startswith(prefix):
            name = label + name[len(prefix):]
            break

    name = name.replace("_", " ").replace("  ", " ").strip()
    name = name.replace("Tomato Tomato", "Tomato")
    return name


def get_treatments(class_name: str, is_healthy: bool) -> list[str]:
    if is_healthy:
        return HEALTHY_TREATMENT
    return DEFAULT_TREATMENT


# ==================== Model Loading ====================
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = None
MODEL_READY = False

try:
    model = resnet18(weights=None, num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    MODEL_READY = True
    logger.info("ResNet18 model loaded successfully.")
except Exception as e:
    logger.error("Model loading failed: %s", e, exc_info=True)


# ==================== Image Validation And Prediction ====================
def check_crop_image(image: Image.Image, global_predictions: list[dict] | None = None) -> dict:
    """Reject obvious non-crop images before returning a disease result."""
    top_class = global_predictions[0]["class_name"] if global_predictions else ""
    top_confidence = global_predictions[0]["confidence"] if global_predictions else 0.0

    if top_class == BG_CLASS_NAME:
        return {
            "is_crop": False,
            "reason": "The uploaded image is not a crop image.",
            "predicted_class": top_class,
        }

    try:
        sample = image.convert("RGB")
        sample.thumbnail((256, 256))
        rgb = np.asarray(sample).astype(np.float32) / 255.0
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return {
                "is_crop": False,
                "reason": "The image cannot be analyzed as a normal RGB image.",
                "predicted_class": top_class,
            }

        hsv = np.asarray(sample.convert("HSV")).astype(np.float32)
        hue = hsv[..., 0]
        saturation = hsv[..., 1] / 255.0
        value = hsv[..., 2] / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        valid_light = (value > 0.10) & (value < 0.98)
        green_leaf = (
            (hue >= 45) & (hue <= 120) &
            (saturation > 0.18) & valid_light &
            (g >= r * 0.82) & (g >= b * 0.72)
        )
        yellow_brown_leaf = (
            (hue >= 20) & (hue <= 50) &
            (saturation > 0.20) & valid_light &
            (g >= b * 0.70)
        )
        plant_ratio = float(np.mean(green_leaf | yellow_brown_leaf))
        green_ratio = float(np.mean(green_leaf))

        gray = rgb.mean(axis=2)
        texture_x = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0
        texture_y = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0
        texture_score = float((texture_x + texture_y) / 2)

        looks_like_crop = (
            green_ratio >= 0.025
            or (plant_ratio >= 0.18 and texture_score >= 0.025)
            or (top_confidence >= 0.70 and plant_ratio >= 0.08)
        )

        return {
            "is_crop": bool(looks_like_crop),
            "reason": "" if looks_like_crop else "The uploaded image does not look like a crop leaf or crop plant.",
            "predicted_class": top_class,
            "plant_ratio": round(plant_ratio, 4),
            "green_ratio": round(green_ratio, 4),
        }
    except Exception as e:
        logger.warning("Crop image validation failed: %s", e)
        return {
            "is_crop": False,
            "reason": "The uploaded image could not be validated.",
            "predicted_class": top_class,
        }


def predict_deep(image: Image.Image, top_k: int = 5, allowed_class_ids: list[int] | None = None) -> list[dict]:
    if not MODEL_READY or model is None:
        return []

    img_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)

        if allowed_class_ids:
            class_ids_tensor = torch.tensor(allowed_class_ids, dtype=torch.long, device=DEVICE)
            allowed_probs = probs[:, class_ids_tensor]
            top_probs, top_positions = torch.topk(allowed_probs, min(top_k, len(allowed_class_ids)))
            top_indices = class_ids_tensor[top_positions]
        else:
            top_probs, top_indices = torch.topk(probs, min(top_k, len(CLASS_NAMES)))

    results = []
    for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
        class_id = int(idx)
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        results.append({
            "class_id": class_id,
            "class_name": class_name,
            "disease": display_class_name(class_name),
            "crop": class_to_crop(class_name),
            "confidence": float(prob),
        })
    return results


def choose_best_crop_prediction(predictions: list[dict]) -> dict | None:
    if not predictions:
        return None

    best = predictions[0]
    if "healthy" not in best["class_name"].lower():
        return best

    disease_candidates = [
        item for item in predictions
        if "healthy" not in item["class_name"].lower() and item["class_name"] != BG_CLASS_NAME
    ]
    if not disease_candidates:
        return best

    best_disease = disease_candidates[0]
    healthy_conf = best["confidence"]
    disease_conf = best_disease["confidence"]

    # If a disease class is reasonably close to the healthy class, prefer disease.
    # This reduces false "healthy" outputs for visibly diseased leaves.
    if disease_conf >= 0.15 and (healthy_conf - disease_conf) <= 0.35:
        return best_disease
    return best


# ==================== Flask Application ====================
app = Flask(__name__)
CORS(app)


@app.route("/api/crops", methods=["GET"])
def get_crops():
    crops = [
        {"name": name, "icon": config["icon"]}
        for name, config in CROP_CONFIG.items()
    ]
    return jsonify({"crops": crops, "total": len(crops)})


@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image file."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Please choose an image file."}), 400

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"Unsupported image format: {ext}. Please use JPG, PNG, BMP, or WEBP."}), 400

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 10 * 1024 * 1024:
        return jsonify({"error": "The image is too large. Please upload an image under 10 MB."}), 400

    selected_crop = normalize_crop_name(request.form.get("crop", "").strip() or "Tomato")
    allowed_class_ids = get_crop_class_ids(selected_crop)
    if not allowed_class_ids:
        return jsonify({"error": "Unsupported crop type. Please select a valid crop."}), 400

    if not MODEL_READY:
        return jsonify({"error": "The model is not ready. Please try again later."}), 503

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        global_predictions = predict_deep(image, top_k=5)
        if not global_predictions:
            return jsonify({"error": "The model could not analyze this image. Please try again."}), 503

        crop_check = check_crop_image(image, global_predictions)
        if not crop_check["is_crop"]:
            return jsonify({
                "error": "This image is not a crop image. Please upload a clear crop leaf or crop plant image.",
                "reason": "non_crop_image",
            }), 422

        crop_predictions = predict_deep(image, top_k=5, allowed_class_ids=allowed_class_ids)
        best = choose_best_crop_prediction(crop_predictions)
        if not best:
            return jsonify({"error": "No valid crop disease result was found. Please upload a clearer crop image."}), 422

        is_healthy = "healthy" in best["class_name"].lower()
        result = {
            "disease": best["disease"],
            "disease_en": best["class_name"],
            "crop": selected_crop,
            "is_healthy": is_healthy,
            "model_type": "ResNet18",
            "treatments": get_treatments(best["class_name"], is_healthy),
        }
        return jsonify(result)
    except Exception as e:
        logger.error("Detection failed: %s", e, exc_info=True)
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model": "ResNet18 45-class crop disease model",
        "device": str(DEVICE),
        "classes": len(CLASS_NAMES),
        "crops": len(CROP_CONFIG),
        "model_ready": MODEL_READY,
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "API endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Crop Disease Detection Backend")
    logger.info("Backend URL: http://127.0.0.1:5000")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
