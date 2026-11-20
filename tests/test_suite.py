"""
tests/test_suite.py
==============================================================================
Automated Verification and Reliability Test Suite
------------------------------------------------------------------------------
Validates syntax, dependencies, validation routines, image annotation,
missing credential error handling, serialization, and secret safety.
==============================================================================
"""

import os
import sys
import io
import json
import unittest
from PIL import Image

# Ensure project root is in sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from services.vision_service import AzureVisionService
from services.local_ocr_service import format_easyocr_results
from utils.image_utils import (
    validate_image_file,
    annotate_objects_on_image,
    annotate_ocr_lines_on_image,
    image_to_bytes,
    generate_sample_assets_if_missing
)


class TestEnvironmentAndImports(unittest.TestCase):
    """Verifies dependencies and package integrity."""

    def test_imports(self):
        """Ensure all required libraries import successfully."""
        import streamlit
        import PIL
        import dotenv
        import requests
        import pandas
        import azure.ai.vision.imageanalysis
        self.assertIsNotNone(streamlit.__version__)
        self.assertIsNotNone(PIL.__version__)


class TestImageValidation(unittest.TestCase):
    """Tests image validation logic against positive and negative test cases."""

    def setUp(self):
        # Create a valid test JPEG in memory
        self.valid_img = Image.new("RGB", (200, 200), color=(10, 20, 30))
        self.valid_bytes = io.BytesIO()
        self.valid_img.save(self.valid_bytes, format="JPEG")
        self.valid_bytes.seek(0)

    def test_valid_jpeg(self):
        """Valid JPEG image must pass validation."""
        is_valid, err, meta = validate_image_file(self.valid_bytes, "test_photo.jpg")
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertEqual(meta["format"], "JPEG")
        self.assertEqual(meta["width"], 200)
        self.assertEqual(meta["height"], 200)

    def test_invalid_extension(self):
        """Disallowed file extensions must be rejected."""
        dummy_file = io.BytesIO(b"some text content")
        is_valid, err, _ = validate_image_file(dummy_file, "notes.txt")
        self.assertFalse(is_valid)
        self.assertIn("Unsupported file extension", err)

    def test_empty_file(self):
        """Zero-byte file must be rejected."""
        empty_file = io.BytesIO(b"")
        is_valid, err, _ = validate_image_file(empty_file, "empty.png")
        self.assertFalse(is_valid)
        self.assertIn("empty", err.lower())

    def test_undersized_image(self):
        """Images smaller than 50x50 must be rejected."""
        tiny_img = Image.new("RGB", (30, 30), color=(0, 0, 0))
        tiny_bytes = io.BytesIO()
        tiny_img.save(tiny_bytes, format="PNG")
        tiny_bytes.seek(0)
        is_valid, err, _ = validate_image_file(tiny_bytes, "tiny.png")
        self.assertFalse(is_valid)
        self.assertIn("too small", err.lower())


class TestImageAnnotation(unittest.TestCase):
    """Tests Pillow-based bounding box and annotation features."""

    def setUp(self):
        self.test_img = Image.new("RGB", (400, 400), color=(50, 50, 50))
        self.sample_objects = [
            {
                "name": "car",
                "confidence": 0.95,
                "confidence_pct": 95.0,
                "bounding_box": {"x": 50, "y": 50, "width": 150, "height": 100}
            },
            {
                "name": "pedestrian",
                "confidence": 0.40,
                "confidence_pct": 40.0,
                "bounding_box": {"x": 220, "y": 60, "width": 60, "height": 180}
            }
        ]

    def test_bounding_box_drawing(self):
        """Bounding boxes must render and produce non-empty image bytes."""
        annotated = annotate_objects_on_image(
            image=self.test_img,
            detected_objects=self.sample_objects,
            min_confidence=0.50
        )
        self.assertEqual(annotated.size, (400, 400))
        img_bytes = image_to_bytes(annotated, "PNG")
        self.assertGreater(len(img_bytes), 0)

    def test_ocr_line_highlighting(self):
        """OCR line highlight rendering must succeed."""
        sample_lines = [
            {"text": "Sample", "bounding_polygon": [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 50, "y": 30}, {"x": 10, "y": 30}]}
        ]
        ann = annotate_ocr_lines_on_image(self.test_img, sample_lines)
        self.assertEqual(ann.size, (400, 400))

    def test_easyocr_results_are_normalized(self):
        """EasyOCR output becomes line text, confidence, counts, and polygons."""
        result = format_easyocr_results([
            ([[10.5, 12.0], [90.0, 12.0], [90.0, 32.0], [10.5, 32.0]], "Hello world", 0.875)
        ])

        self.assertTrue(result["detected"])
        self.assertEqual(result["total_lines"], 1)
        self.assertEqual(result["total_words"], 2)
        self.assertEqual(result["full_text"], "Hello world")
        self.assertEqual(result["lines"][0]["confidence_pct"], 87.5)
        self.assertEqual(result["lines"][0]["bounding_polygon"][0], {"x": 10, "y": 12})

    def test_easyocr_empty_results(self):
        """Empty EasyOCR output is represented as a successful no-text result."""
        result = format_easyocr_results([])

        self.assertFalse(result["detected"])
        self.assertEqual(result["total_lines"], 0)
        self.assertEqual(result["total_words"], 0)
        self.assertEqual(result["full_text"], "")


class TestVisionServiceAndDemoHandling(unittest.TestCase):
    """Tests configuration checking, missing credential handling, and demo presets."""

    def test_unconfigured_service(self):
        """Unconfigured service must report not configured and refuse fake calls."""
        service = AzureVisionService(endpoint="", key="")
        self.assertFalse(service.is_configured())
        self.assertEqual(service.get_masked_key(), "Not Configured")

        # Live call attempt must return graceful error without raising or faking
        res = service.analyze_image(b"fake_image_bytes")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "CONFIGURATION_ERROR")

    def test_placeholder_credential_rejection(self):
        """Placeholder credentials must be flagged as unconfigured."""
        service = AzureVisionService(
            endpoint="https://your-resource-name.cognitiveservices.azure.com/",
            key="your_azure_vision_key_here"
        )
        self.assertTrue(service.is_placeholder())
        self.assertFalse(service.is_configured())
        is_valid, msg = service.validate_configuration()
        self.assertFalse(is_valid)

    def test_demo_presets(self):
        """Demo presets must return valid structured data explicitly flagged as demo."""
        for preset in ["street", "presentation", "document"]:
            demo = AzureVisionService.get_sample_demo_analysis(preset)
            self.assertTrue(demo["success"])
            self.assertTrue(demo["is_demo"])
            self.assertEqual(demo["source"], "demo_offline_sample")
            self.assertIn("caption", demo)
            self.assertIn("tags", demo)
            self.assertIn("objects", demo)
            self.assertIn("ocr", demo)
            self.assertGreater(len(demo["tags"]), 0)


class TestSecretScanner(unittest.TestCase):
    """Scans all repository files to ensure no sensitive Azure keys are hardcoded."""

    def test_no_hardcoded_secrets(self):
        # Precise regex patterns matching genuine active API key shapes
        suspicious_patterns = [
            r"sk-[a-zA-Z0-9_-]{20,}",
            r"Ocp-Apim-Subscription-Key:\s*[0-9a-fA-F]{32}",
            r"AZURE_VISION_KEY=[0-9a-fA-F]{32}"
        ]

        for root, _, files in os.walk(PROJECT_DIR):
            # Skip git, tests, cache, and virtual environments
            if any(skip in root for skip in [".git", "__pycache__", "venv", ".venv", "tests"]):
                continue
            for fname in files:
                if fname.endswith((".py", ".json", ".md", ".env", ".example")):
                    fpath = os.path.join(root, fname)
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        for pat in suspicious_patterns:
                            import re
                            match = re.search(pat, content)
                            self.assertIsNone(
                                match,
                                f"Possible sensitive credential found in {fname}: {match}"
                            )


if __name__ == "__main__":
    unittest.main()
