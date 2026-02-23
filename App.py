import os
import io
import base64
import logging
import time
import requests
from functools import wraps
from flask import Flask, request, jsonify
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

app = Flask(__name__)

# =========================
# CONFIG
# =========================

MAX_IMAGE_MB = 15
MAX_WIDTH = 4096
MAX_HEIGHT = 4096
RATE_LIMIT_SECONDS = 2

API_KEY = os.environ.get("API_KEY")
SIGHTENGINE_USER = os.environ.get("SIGHTENGINE_USER")
SIGHTENGINE_SECRET = os.environ.get("SIGHTENGINE_SECRET")

request_times = {}

logging.basicConfig(level=logging.INFO)

# =========================
# SECURITY
# =========================

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("x-api-key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def rate_limit(ip):
    now = time.time()
    if ip in request_times:
        if now - request_times[ip] < RATE_LIMIT_SECONDS:
            return False
    request_times[ip] = now
    return True


# =========================
# VALIDATION
# =========================

def validate_image(image_bytes):
    if len(image_bytes) > MAX_IMAGE_MB * 1024 * 1024:
        return False, "Image too large"

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except:
        return False, "Invalid image"

    return True, None


# =========================
# AI MODERATION
# =========================

def ai_moderation(image_bytes):

    response = requests.post(
        "https://api.sightengine.com/1.0/check.json",
        data={
            "models": "nudity-2.0",
            "api_user": SIGHTENGINE_USER,
            "api_secret": SIGHTENGINE_SECRET
        },
        files={
            "media": image_bytes
        }
    )

    result = response.json()

    if "nudity" in result:
        if result["nudity"]["sexual_activity"] > 0.5:
            return False
        if result["nudity"]["sexual_display"] > 0.5:
            return False

    return True


# =========================
# PROCESSING
# =========================

def process_image(image, mode):

    image = image.convert("RGB")

    width, height = image.size
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        image.thumbnail((MAX_WIDTH, MAX_HEIGHT))

    if mode == "grayscale":
        return image.convert("L")

    if mode == "edge":
        return image.filter(ImageFilter.FIND_EDGES)

    if mode == "blur":
        return image.filter(ImageFilter.GaussianBlur(4))

    if mode == "sharpen":
        return image.filter(ImageFilter.SHARPEN)

    if mode == "contrast":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(2.0)

    return image


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return "AI Protected Image Processor Running"


@app.route("/process", methods=["POST"])
@require_api_key
def process():

    ip = request.remote_addr

    if not rate_limit(ip):
        return jsonify({"error": "Too many requests"}), 429

    try:
        data = request.json

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        mode = data.get("mode", "grayscale")

        image_bytes = base64.b64decode(data["image"])

        valid, error = validate_image(image_bytes)
        if not valid:
            return jsonify({"error": error}), 400

        # 🔐 AI moderation
        if not ai_moderation(image_bytes):
            return jsonify({"error": "Inappropriate content detected"}), 403

        image = Image.open(io.BytesIO(image_bytes))

        processed = process_image(image, mode)

        buffer = io.BytesIO()
        processed.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify({
            "status": "success",
            "mode": mode,
            "image": encoded
        })

    except Exception as e:
        logging.exception("Server error")
        return jsonify({"error": "Processing failed"}), 500


if __name__ == "__main__":
    app.run()
