"""
utils/image_utils.py
==============================================================================
Image Processing and Annotation Utility Module
------------------------------------------------------------------------------
Provides robust image validation, metadata extraction, dynamic bounding box
annotation using Pillow, and high-quality sample asset generation.
==============================================================================
"""

import io
import os
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont


# Supported image types and constraints (aligned with Azure AI Vision 4.0 limits)
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
SUPPORTED_FORMATS = ["JPEG", "PNG", "JPG"]
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB max
MIN_DIMENSION = 50                     # 50 pixels min width/height
MAX_DIMENSION = 16000                  # 16,000 pixels max width/height

# Distinct high-contrast color palette for bounding box visualization
PALETTE = [
    (0, 183, 195),    # Cyan / Azure teal
    (16, 185, 129),   # Emerald green
    (245, 158, 11),   # Amber
    (239, 68, 68),    # Coral red
    (139, 92, 246),   # Electric violet
    (14, 165, 233),   # Azure sky
    (249, 115, 22),   # Sunset orange
    (236, 72, 153),   # Magenta pink
]


def validate_image_file(file_obj, filename: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validates the uploaded file object for extension, MIME/format, file size,
    and image dimensions.

    Returns:
        (is_valid, error_message, metadata_dict)
    """
    if file_obj is None:
        return False, "No file provided.", {}

    # 1. Extension check
    ext = os.path.splitext(filename.lower())[1]
    if ext not in SUPPORTED_EXTENSIONS:
        return False, (
            f"Unsupported file extension '{ext}'. "
            f"Only JPG, JPEG, and PNG images are supported."
        ), {}

    # 2. File size check
    # Check if file has getbuffer() or read()
    if hasattr(file_obj, "getbuffer"):
        raw_bytes = bytes(file_obj.getbuffer())
    elif hasattr(file_obj, "getvalue"):
        raw_bytes = file_obj.getvalue()
    elif isinstance(file_obj, bytes):
        raw_bytes = file_obj
    else:
        file_obj.seek(0)
        raw_bytes = file_obj.read()

    file_size_bytes = len(raw_bytes)
    if file_size_bytes == 0:
        return False, "Uploaded file is empty (0 bytes).", {}

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = file_size_bytes / (1024 * 1024)
        return False, (
            f"File size ({size_mb:.2f} MB) exceeds the maximum allowed limit of 20 MB."
        ), {}

    # 3. Format and dimension check via Pillow
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img_format = img.format.upper() if img.format else "UNKNOWN"
        width, height = img.size

        if img_format not in SUPPORTED_FORMATS:
            return False, (
                f"Image format '{img_format}' is not supported. "
                f"Please upload a valid JPEG or PNG image."
            ), {}

        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            return False, (
                f"Image dimensions ({width}x{height} px) are too small. "
                f"Azure AI Vision requires images of at least {MIN_DIMENSION}x{MIN_DIMENSION} pixels."
            ), {}

        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            return False, (
                f"Image dimensions ({width}x{height} px) exceed maximum supported {MAX_DIMENSION}px."
            ), {}

        # Calculate readable metadata
        aspect_ratio = round(width / height, 2) if height > 0 else 1.0
        file_size_kb = round(file_size_bytes / 1024, 2)
        file_size_str = (
            f"{file_size_kb:.1f} KB" if file_size_kb < 1024
            else f"{file_size_kb / 1024:.2f} MB"
        )

        metadata = {
            "filename": filename,
            "format": img_format,
            "width": width,
            "height": height,
            "dimensions": f"{width} × {height} px",
            "aspect_ratio": f"{aspect_ratio}:1",
            "mode": img.mode,
            "file_size_bytes": file_size_bytes,
            "file_size_readable": file_size_str
        }

        return True, None, metadata

    except Exception as e:
        return False, f"Failed to decode image file: {str(e)}", {}


def annotate_objects_on_image(
    image: Image.Image,
    detected_objects: List[Dict[str, Any]],
    min_confidence: float = 0.0,
    show_labels: bool = True,
    show_confidence: bool = True,
    line_thickness: int = 3
) -> Image.Image:
    """
    Renders visual bounding boxes and styled identification badges
    onto the image using Pillow.

    Args:
        image: Original PIL Image.
        detected_objects: List of objects with 'name', 'confidence', 'bounding_box'.
        min_confidence: Filter threshold (0.0 to 1.0).
        show_labels: Whether to display text labels.
        show_confidence: Whether to display confidence percentage in badge.
        line_thickness: Stroke width for bounding box.

    Returns:
        Annotated PIL Image (copy).
    """
    # Create RGBA working image for alpha blending
    working_img = image.convert("RGBA")
    overlay = Image.new("RGBA", working_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Attempt to load default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    img_w, img_h = working_img.size

    for idx, obj in enumerate(detected_objects):
        conf = float(obj.get("confidence", 0.0))
        if conf < min_confidence:
            continue

        name = obj.get("name", "Object")
        bb = obj.get("bounding_box", {})

        x = int(bb.get("x", 0))
        y = int(bb.get("y", 0))
        w = int(bb.get("width", 0))
        h = int(bb.get("height", 0))

        # Clamp coordinates to image boundaries
        x1 = max(0, min(x, img_w - 1))
        y1 = max(0, min(y, img_h - 1))
        x2 = max(0, min(x + w, img_w - 1))
        y2 = max(0, min(y + h, img_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = PALETTE[idx % len(PALETTE)]
        box_color = (*color, 255)
        fill_color = (*color, 35)  # Subtle translucent box fill

        # 1. Translucent box fill
        draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        # 2. Outer bounding box outline with requested thickness
        for t in range(line_thickness):
            draw.rectangle(
                [x1 - t, y1 - t, x2 + t, y2 + t],
                outline=box_color
            )

        # 3. Label Badge
        if show_labels:
            conf_str = f" {conf * 100:.1f}%" if show_confidence else ""
            label_text = f"{name.upper()}{conf_str}"

            # Calculate label text size
            if font and hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            else:
                text_w = len(label_text) * 7
                text_h = 12

            pad = 4
            badge_w = text_w + (pad * 2)
            badge_h = text_h + (pad * 2)

            # Position badge above box if space permits, else inside top-left
            if y1 - badge_h >= 0:
                badge_y1 = y1 - badge_h
                badge_y2 = y1
            else:
                badge_y1 = y1
                badge_y2 = y1 + badge_h

            badge_x1 = x1
            badge_x2 = min(x1 + badge_w, img_w - 1)

            # Draw badge background
            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=box_color)

            # Draw badge text (white text with slight shadow)
            text_x = badge_x1 + pad
            text_y = badge_y1 + pad
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255, 255), font=font)

    # Composite overlay onto working image
    composite = Image.alpha_composite(working_img, overlay)
    return composite.convert("RGB")


def annotate_ocr_lines_on_image(
    image: Image.Image,
    ocr_lines: List[Dict[str, Any]],
    line_color: Tuple[int, int, int] = (0, 183, 195)
) -> Image.Image:
    """
    Overlays subtle highlight polygons around detected OCR text lines.
    """
    working_img = image.convert("RGBA")
    overlay = Image.new("RGBA", working_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for line in ocr_lines:
        polygon = line.get("bounding_polygon", [])
        if len(polygon) >= 4:
            pts = [(p["x"], p["y"]) for p in polygon if "x" in p and "y" in p]
            if len(pts) >= 4:
                draw.polygon(pts, fill=(*line_color, 45), outline=(*line_color, 200))

    composite = Image.alpha_composite(working_img, overlay)
    return composite.convert("RGB")


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Encodes a PIL Image into bytes buffer."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def generate_sample_assets_if_missing(assets_dir: str):
    """
    Programmatically creates high-quality, lightweight sample images for the
    three offline demonstration presets if they do not already exist on disk:
    1. sample_street.jpg (Urban Traffic Scene)
    2. sample_presentation.png (Technical AI Architecture Slide)
    3. sample_invoice.png (Business Invoice Document)
    """
    os.makedirs(assets_dir, exist_ok=True)

    street_path = os.path.join(assets_dir, "sample_street.jpg")
    pres_path = os.path.join(assets_dir, "sample_presentation.png")
    inv_path = os.path.join(assets_dir, "sample_invoice.png")

    font = ImageFont.load_default()

    # 1. Sample Street Scene
    if not os.path.exists(street_path):
        img = Image.new("RGB", (800, 550), color=(30, 41, 59))  # Dark slate background
        draw = ImageDraw.Draw(img)

        # Draw sky and buildings
        draw.rectangle([0, 0, 800, 240], fill=(20, 29, 45))     # Deep night sky
        # Buildings
        draw.rectangle([30, 80, 160, 240], fill=(45, 55, 72))
        draw.rectangle([180, 50, 320, 240], fill=(55, 65, 85))
        draw.rectangle([350, 100, 520, 240], fill=(40, 50, 70))
        draw.rectangle([550, 70, 750, 240], fill=(50, 60, 80))

        # Windows
        for bx in [50, 200, 370, 570]:
            for by in range(110, 220, 30):
                draw.rectangle([bx, by, bx + 20, by + 18], fill=(255, 230, 120))

        # Road
        draw.rectangle([0, 240, 800, 550], fill=(40, 44, 52))    # Asphalt
        draw.rectangle([0, 370, 800, 385], fill=(255, 255, 255)) # Lane divider

        # Crosswalk
        for cx in range(80, 720, 50):
            draw.rectangle([cx, 480, cx + 30, 530], fill=(220, 220, 220))

        # Cars (Rectangles representing vehicles)
        # Car 1
        draw.rectangle([50, 280, 310, 460], fill=(59, 130, 246), outline=(255, 255, 255), width=2)
        draw.text((70, 350), "BLUE SEDAN CAR", fill=(255, 255, 255), font=font)

        # Car 2
        draw.rectangle([340, 290, 580, 460], fill=(239, 68, 68), outline=(255, 255, 255), width=2)
        draw.text((360, 350), "RED SUV CAR", fill=(255, 255, 255), font=font)

        # Pedestrian
        draw.rectangle([620, 240, 710, 460], fill=(16, 185, 129), outline=(255, 255, 255), width=2)
        draw.text((630, 340), "PEDESTRIAN", fill=(255, 255, 255), font=font)

        # Street signs with text
        draw.rectangle([170, 180, 250, 205], fill=(15, 118, 110))
        draw.text((175, 185), "MAIN ST", fill=(255, 255, 255), font=font)

        draw.rectangle([680, 140, 780, 190], fill=(255, 255, 255))
        draw.text((685, 155), "SPEED LIMIT 25", fill=(0, 0, 0), font=font)

        img.save(street_path, format="JPEG", quality=90)

    # 2. Sample Presentation Slide
    if not os.path.exists(pres_path):
        img = Image.new("RGB", (800, 550), color=(15, 23, 42))  # Near black / navy
        draw = ImageDraw.Draw(img)

        # Slide header banner
        draw.rectangle([0, 0, 800, 90], fill=(30, 58, 138))
        draw.text((40, 32), "Microsoft Azure AI Fundamentals (AI-900)", fill=(255, 255, 255), font=font)

        # Subtitle
        draw.text((40, 115), "Computer Vision Workloads & Core Capabilities:", fill=(56, 189, 248), font=font)

        # Bullets
        bullets = [
            "1. Image Classification & Semantic Tagging",
            "2. Object Detection with Bounding Box Coordinates",
            "3. Optical Character Recognition (OCR Read API)",
            "4. Natural Language Image Caption Generation",
            "5. Responsible AI: Fairness, Reliability & Privacy"
        ]
        y_pos = 160
        for b in bullets:
            draw.rectangle([40, y_pos - 4, 48, y_pos + 4], fill=(56, 189, 248))
            draw.text((60, y_pos - 6), b, fill=(241, 245, 249), font=font)
            y_pos += 45

        # Architecture diagram mockup box on right
        draw.rectangle([500, 140, 750, 480], fill=(30, 41, 59), outline=(56, 189, 248), width=2)
        draw.text((520, 160), "[CLOUD AI ARCHITECTURE]", fill=(56, 189, 248), font=font)
        draw.rectangle([530, 200, 720, 250], fill=(15, 23, 42), outline=(255, 255, 255))
        draw.text((545, 220), "Client: Streamlit App", fill=(255, 255, 255), font=font)
        draw.rectangle([530, 290, 720, 340], fill=(15, 23, 42), outline=(59, 130, 246))
        draw.text((540, 310), "REST / Python SDK 4.0", fill=(56, 189, 248), font=font)
        draw.rectangle([530, 380, 720, 440], fill=(24, 76, 120), outline=(0, 183, 195))
        draw.text((540, 400), "Azure AI Vision Service", fill=(255, 255, 255), font=font)

        img.save(pres_path, format="PNG")

    # 3. Sample Business Invoice
    if not os.path.exists(inv_path):
        img = Image.new("RGB", (800, 550), color=(250, 250, 252))  # Clean off-white paper
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([0, 0, 800, 70], fill=(15, 23, 42))
        draw.text((40, 25), "INVOICE #INV-2026-0891", fill=(255, 255, 255), font=font)
        draw.text((550, 25), "Cloud Tech Solutions Pvt Ltd", fill=(148, 163, 184), font=font)

        # Details
        draw.text((40, 90), "Billed To: Cloud Tech Solutions Pvt Ltd", fill=(15, 23, 42), font=font)
        draw.text((40, 115), "Date of Issue: November 20, 2026", fill=(100, 116, 139), font=font)
        draw.text((40, 140), "Payment Status: PAID ONLINE", fill=(16, 185, 129), font=font)

        # Table header
        draw.rectangle([40, 180, 760, 215], fill=(226, 232, 240))
        draw.text((55, 192), "DESCRIPTION", fill=(15, 23, 42), font=font)
        draw.text((450, 192), "QTY", fill=(15, 23, 42), font=font)
        draw.text((650, 192), "AMOUNT", fill=(15, 23, 42), font=font)

        # Table rows
        rows = [
            ("Item 1: Azure AI Vision Cognitive API", "1", "$120.00"),
            ("Item 2: Cloud Storage Hot Tier", "1", "$35.00"),
            ("Item 3: SSL Security Certificate", "1", "$15.50"),
        ]
        ry = 230
        for desc, qty, amt in rows:
            draw.text((55, ry), desc, fill=(51, 65, 85), font=font)
            draw.text((460, ry), qty, fill=(51, 65, 85), font=font)
            draw.text((650, ry), amt, fill=(51, 65, 85), font=font)
            draw.line([40, ry + 25, 760, ry + 25], fill=(226, 232, 240))
            ry += 40

        # Totals
        draw.rectangle([400, 370, 760, 480], fill=(241, 245, 249), outline=(203, 213, 225))
        draw.text((430, 390), "Subtotal: $155.00 | Tax (10%): $15.50", fill=(71, 85, 105), font=font)
        draw.text((430, 430), "Total Due: $170.50 USD", fill=(15, 23, 42), font=font)

        img.save(inv_path, format="PNG")
