import os
import io
import base64
import logging
import time
import requests
from functools import wraps
from typing import Tuple, Optional, Dict, Any
from flask import Flask, request, jsonify
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from datetime import datetime, timedelta
from werkzeug.exceptions import RequestEntityTooLarge
import hashlib

app = Flask(__name__)

# =========================
# CONFIG
# =========================

MAX_IMAGE_MB = 15
MAX_WIDTH = 4096
MAX_HEIGHT = 4096
RATE_LIMIT_SECONDS = 2
RATE_LIMIT_MAX_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour

API_KEY = os.environ.get("API_KEY")
SIGHTENGINE_USER = os.environ.get("SIGHTENGINE_USER")
SIGHTENGINE_SECRET = os.environ.get("SIGHTENGINE_SECRET")
SIGHTENGINE_TIMEOUT = 10

# Cache settings
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "True").lower() == "true"
CACHE_EXPIRY = 3600  # 1 hour

request_times = {}
processing_cache = {}
moderation_cache = {}

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported image modes
SUPPORTED_MODES = {
    "grayscale",
    "edge",
    "blur",
    "sharpen",
    "contrast",
    "invert",
    "brightness",
    "saturation"
}

# =========================
# SECURITY
# =========================

def require_api_key(f):
    """Decorator to validate API key from request headers."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY:
            logger.error("API_KEY not configured")
            return jsonify({"error": "Server misconfiguration"}), 500
        
        key = request.headers.get("x-api-key")
        if not key:
            logger.warning(f"Missing API key from {request.remote_addr}")
            return jsonify({"error": "Missing API key"}), 401
        
        if key != API_KEY:
            logger.warning(f"Invalid API key attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
        
        return f(*args, **kwargs)
    return decorated


def rate_limit(ip: str) -> bool:
    """
    Implement sliding window rate limiting.
    
    Args:
        ip: Client IP address
        
    Returns:
        True if request is allowed, False if rate limit exceeded
    """
    now = time.time()
    
    if ip not in request_times:
        request_times[ip] = []
    
    # Remove old requests outside the window
    request_times[ip] = [
        t for t in request_times[ip] 
        if now - t < RATE_LIMIT_WINDOW
    ]
    
    # Check if over limit
    if len(request_times[ip]) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning(f"Rate limit exceeded for IP: {ip}")
        return False
    
    # Also check burst limit
    recent = [t for t in request_times[ip] if now - t < RATE_LIMIT_SECONDS]
    if len(recent) >= 3:  # Max 3 requests per 2 seconds
        return False
    
    request_times[ip].append(now)
    return True


def get_client_ip() -> str:
    """
    Get client IP address, accounting for proxies.
    
    Returns:
        Client IP address
    """
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        return request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    return request.remote_addr or "unknown"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal attacks."""
    import re
    # Remove any path separators and suspicious characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    return filename[:255]  # Limit length

# =========================
# VALIDATION
# =========================

def validate_image(image_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Validate image format, size, and integrity.
    
    Args:
        image_bytes: Raw image data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check size
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_MB:
        logger.warning(f"Image too large: {size_mb:.2f}MB")
        return False, f"Image too large: maximum {MAX_IMAGE_MB}MB allowed"
    
    if size_mb == 0:
        return False, "Empty image data"
    
    # Validate image format
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Verify the image format
        if img.format not in ['JPEG', 'PNG', 'GIF', 'WEBP', 'BMP']:
            return False, f"Unsupported format: {img.format}"
        
        # Check dimensions
        width, height = img.size
        if width == 0 or height == 0:
            return False, "Invalid image dimensions"
        
        if width > MAX_WIDTH or height > MAX_HEIGHT:
            return False, f"Image dimensions exceed limits ({MAX_WIDTH}x{MAX_HEIGHT})"
        
        # Verify image integrity
        img.verify()
        
    except Image.UnidentifiedImageError:
        logger.warning("Invalid image format")
        return False, "Invalid or corrupted image"
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return False, "Image validation failed"
    
    return True, None


def get_image_hash(image_bytes: bytes) -> str:
    """Generate SHA256 hash of image for caching."""
    return hashlib.sha256(image_bytes).hexdigest()

# =========================
# AI MODERATION
# =========================

def ai_moderation(image_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Check image content using SightEngine API.
    
    Args:
        image_bytes: Raw image data
        
    Returns:
        Tuple of (is_safe, error_message)
    """
    if not SIGHTENGINE_USER or not SIGHTENGINE_SECRET:
        logger.warning("SightEngine credentials not configured, skipping moderation")
        return True, None
    
    # Check cache
    image_hash = get_image_hash(image_bytes)
    if CACHE_ENABLED and image_hash in moderation_cache:
        cached_result, timestamp = moderation_cache[image_hash]
        if time.time() - timestamp < CACHE_EXPIRY:
            logger.info("Using cached moderation result")
            return cached_result, None
    
    try:
        response = requests.post(
            "https://api.sightengine.com/1.0/check.json",
            data={
                "models": "nudity-2.0,offensive,gore",
                "api_user": SIGHTENGINE_USER,
                "api_secret": SIGHTENGINE_SECRET
            },
            files={
                "media": io.BytesIO(image_bytes)
            },
            timeout=SIGHTENGINE_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.error(f"SightEngine API error: {response.status_code}")
            return False, "Content moderation service unavailable"
        
        result = response.json()
        
        # Check for errors in API response
        if "error" in result:
            logger.error(f"SightEngine error: {result['error']}")
            return False, "Moderation check failed"
        
        # Evaluate nudity
        if "nudity" in result:
            nudity = result["nudity"]
            if nudity.get("sexual_activity", 0) > 0.5:
                logger.warning("Inappropriate content detected (sexual_activity)")
                cache_moderation_result(image_hash, False)
                return False, "Inappropriate content detected"
            if nudity.get("sexual_display", 0) > 0.5:
                logger.warning("Inappropriate content detected (sexual_display)")
                cache_moderation_result(image_hash, False)
                return False, "Inappropriate content detected"
        
        # Evaluate gore
        if "gore" in result:
            if result["gore"].get("gore", 0) > 0.7:
                logger.warning("Inappropriate content detected (gore)")
                cache_moderation_result(image_hash, False)
                return False, "Inappropriate content detected"
        
        # Evaluate offensive content
        if "offensive" in result:
            if result["offensive"].get("offensive", 0) > 0.8:
                logger.warning("Inappropriate content detected (offensive)")
                cache_moderation_result(image_hash, False)
                return False, "Inappropriate content detected"
        
        cache_moderation_result(image_hash, True)
        return True, None
        
    except requests.Timeout:
        logger.error("SightEngine API timeout")
        return False, "Moderation service timeout"
    except requests.RequestException as e:
        logger.error(f"SightEngine API error: {str(e)}")
        return False, "Moderation service error"


def cache_moderation_result(image_hash: str, result: bool):
    """Cache moderation result with timestamp."""
    moderation_cache[image_hash] = (result, time.time())
    
    # Clean old cache entries
    if len(moderation_cache) > 1000:
        oldest_key = min(
            moderation_cache.keys(),
            key=lambda k: moderation_cache[k][1]
        )
        del moderation_cache[oldest_key]

# =========================
# PROCESSING
# =========================

def process_image(image: Image.Image, mode: str) -> Tuple[Image.Image, Optional[str]]:
    """
    Apply image processing based on mode.
    
    Args:
        image: PIL Image object
        mode: Processing mode
        
    Returns:
        Tuple of (processed_image, error_message)
    """
    try:
        # Validate mode
        if mode not in SUPPORTED_MODES:
            return None, f"Unsupported mode: {mode}"
        
        # Convert to RGB for processing
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        else:
            image = image.convert("RGB")
        
        # Resize if necessary
        width, height = image.size
        if width > MAX_WIDTH or height > MAX_HEIGHT:
            image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)
        
        # Apply processing based on mode
        if mode == "grayscale":
            return image.convert("L"), None
        
        elif mode == "edge":
            return image.filter(ImageFilter.FIND_EDGES), None
        
        elif mode == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=4)), None
        
        elif mode == "sharpen":
            return image.filter(ImageFilter.SHARPEN), None
        
        elif mode == "contrast":
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(2.0), None
        
        elif mode == "invert":
            if image.mode == "L":
                return ImageEnhance.Color(image).enhance(0), None
            return ImageOps.invert(image.convert("RGB")), None
        
        elif mode == "brightness":
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(1.2), None
        
        elif mode == "saturation":
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(1.5), None
        
        return image, None
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None, f"Processing failed: {str(e)}"


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> Optional[str]:
    """Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Output format (PNG, JPEG, etc.)
        
    Returns:
        Base64 encoded string or None on error
    """
    try:
        buffer = io.BytesIO()
        
        # Handle format-specific settings
        if format == "JPEG":
            image = image.convert("RGB")
            image.save(buffer, format=format, quality=85, optimize=True)
        else:
            image.save(buffer, format=format, optimize=True)
        
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded
        
    except Exception as e:
        logger.error(f"Image encoding error: {str(e)}")
        return None

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "service": "AI Protected Image Processor",
        "version": "2.0",
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route("/health", methods=["GET"])
def health_check():
    """Detailed health check endpoint."""
    health_status = {
        "status": "healthy",
        "checks": {
            "api_key_configured": bool(API_KEY),
            "sightengine_configured": bool(SIGHTENGINE_USER and SIGHTENGINE_SECRET),
            "cache_enabled": CACHE_ENABLED
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if not API_KEY:
        health_status["status"] = "degraded"
        health_status["warnings"] = ["API_KEY not configured"]
    
    return jsonify(health_status), 200

@app.route("/modes", methods=["GET"])
@require_api_key
def get_supported_modes():
    """Get list of supported processing modes."""
    return jsonify({
        "supported_modes": sorted(list(SUPPORTED_MODES))
    }), 200

@app.route("/process", methods=["POST"])
@require_api_key
def process():
    """
    Main image processing endpoint.
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "mode": "grayscale|edge|blur|sharpen|contrast|invert|brightness|saturation",
        "format": "PNG" (optional, default: PNG)
    }
    """
    client_ip = get_client_ip()
    
    # Rate limiting
    if not rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return jsonify({"error": "Too many requests. Please try again later."}), 429
    
    try:
        # Parse request
        data = request.json
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if "image" not in data:
            return jsonify({"error": "Missing required field: image"}), 400
        
        # Validate mode
        mode = data.get("mode", "grayscale").lower()
        if mode not in SUPPORTED_MODES:
            return jsonify({
                "error": f"Invalid mode. Supported modes: {', '.join(sorted(SUPPORTED_MODES))}"
            }), 400
        
        # Get output format
        output_format = data.get("format", "PNG").upper()
        if output_format not in ["PNG", "JPEG", "WEBP"]:
            output_format = "PNG"
        
        # Decode image
        try:
            image_bytes = base64.b64decode(data["image"])
        except Exception as e:
            logger.error(f"Base64 decode error: {str(e)}")
            return jsonify({"error": "Invalid base64 image encoding"}), 400
        
        # Validate image
        is_valid, validation_error = validate_image(image_bytes)
        if not is_valid:
            logger.warning(f"Image validation failed: {validation_error}")
            return jsonify({"error": validation_error}), 400
        
        # AI moderation
        is_safe, moderation_error = ai_moderation(image_bytes)
        if not is_safe:
            return jsonify({"error": moderation_error or "Inappropriate content detected"}), 403
        
        # Open image
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Image open error: {str(e)}")
            return jsonify({"error": "Failed to open image"}), 400
        
        # Process image
        processed_image, process_error = process_image(image, mode)
        if processed_image is None:
            return jsonify({"error": process_error or "Processing failed"}), 400
        
        # Encode result
        encoded = encode_image_to_base64(processed_image, output_format)
        if not encoded:
            return jsonify({"error": "Failed to encode processed image"}), 500
        
        logger.info(f"Successfully processed image with mode: {mode} for IP: {client_ip}")
        
        return jsonify({
            "status": "success",
            "mode": mode,
            "format": output_format,
            "image": encoded,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except RequestEntityTooLarge:
        return jsonify({"error": "Request too large"}), 413
    except Exception as e:
        logger.exception(f"Unexpected server error for IP {client_ip}")
        return jsonify({"error": "Processing failed. Please try again."}), 500

@app.route("/batch-process", methods=["POST"])
@require_api_key
def batch_process():
    """
    Process multiple images in one request.
    
    Expected JSON payload:
    {
        "images": [
            {"image": "base64_encoded_image", "mode": "grayscale"},
            ...
        ]
    }
    """
    client_ip = get_client_ip()
    
    if not rate_limit(client_ip):
        return jsonify({"error": "Too many requests"}), 429
    
    try:
        data = request.json
        
        if not data or "images" not in data:
            return jsonify({"error": "Missing required field: images"}), 400
        
        images = data["images"]
        
        if not isinstance(images, list):
            return jsonify({"error": "images must be a list"}), 400
        
        if len(images) > 10:
            return jsonify({"error": "Maximum 10 images per batch"}), 400
        
        results = []
        
        for i, img_data in enumerate(images):
            try:
                if not isinstance(img_data, dict) or "image" not in img_data:
                    results.append({
                        "index": i,
                        "status": "error",
                        "error": "Missing image data"
                    })
                    continue
                
                mode = img_data.get("mode", "grayscale").lower()
                
                if mode not in SUPPORTED_MODES:
                    results.append({
                        "index": i,
                        "status": "error",
                        "error": f"Invalid mode: {mode}"
                    })
                    continue
                
                image_bytes = base64.b64decode(img_data["image"])
                
                is_valid, validation_error = validate_image(image_bytes)
                if not is_valid:
                    results.append({
                        "index": i,
                        "status": "error",
                        "error": validation_error
                    })
                    continue
                
                is_safe, _ = ai_moderation(image_bytes)
                if not is_safe:
                    results.append({
                        "index": i,
                        "status": "error",
                        "error": "Inappropriate content detected"
                    })
                    continue
                
                image = Image.open(io.BytesIO(image_bytes))
                processed_image, process_error = process_image(image, mode)
                
                if processed_image is None:
                    results.append({
                        "index": i,
                        "status": "error",
                        "error": process_error
                    })
                    continue
                
                encoded = encode_image_to_base64(processed_image)
                
                results.append({
                    "index": i,
                    "status": "success",
                    "mode": mode,
                    "image": encoded
                })
                
            except Exception as e:
                logger.error(f"Batch processing error at index {i}: {str(e)}")
                results.append({
                    "index": i,
                    "status": "error",
                    "error": "Processing failed"
                })
        
        return jsonify({
            "status": "batch_complete",
            "total": len(images),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.exception("Batch processing error")
        return jsonify({"error": "Batch processing failed"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

# =========================
# CLEANUP
# =========================

def cleanup_old_cache_entries():
    """Periodically clean up old cache entries."""
    now = time.time()
    
    # Clean moderation cache
    expired_keys = [
        k for k, (_, timestamp) in moderation_cache.items()
        if now - timestamp > CACHE_EXPIRY
    ]
    for key in expired_keys:
        del moderation_cache[key]
    
    # Clean request times for IPs with no recent activity
    expired_ips = [
        ip for ip, times in request_times.items()
        if not any(now - t < RATE_LIMIT_WINDOW for t in times)
    ]
    for ip in expired_ips:
        del request_times[ip]


if __name__ == "__main__":
    logger.info("Starting AI Protected Image Processor")
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_ENV") == "development"
    )
