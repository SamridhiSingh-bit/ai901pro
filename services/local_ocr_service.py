"""Local OCR using EasyOCR without cloud credentials."""

from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
from PIL import Image


class LocalOCRError(RuntimeError):
    """Raised when the local OCR engine cannot be initialized or run."""


def format_easyocr_results(results: List[Any]) -> Dict[str, Any]:
    """Convert EasyOCR output into the application's OCR result structure."""
    lines = []
    for bounds, text, confidence in results:
        polygon = [
            {"x": int(point[0]), "y": int(point[1])}
            for point in bounds
        ]
        lines.append({
            "text": text,
            "words_count": len(text.split()),
            "confidence": float(confidence),
            "confidence_pct": round(float(confidence) * 100, 2),
            "bounding_polygon": polygon
        })

    return {
        "detected": bool(lines),
        "total_lines": len(lines),
        "total_words": sum(line["words_count"] for line in lines),
        "lines": lines,
        "full_text": "\n".join(line["text"] for line in lines),
        "engine": "EasyOCR (English, CPU)"
    }


class LocalOCRService:
    """Runs English OCR locally on CPU; model weights are cached by EasyOCR."""

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_reader() -> Any:
        try:
            import easyocr
        except ImportError as exc:
            raise LocalOCRError(
                "EasyOCR is not installed. Install project dependencies with "
                "'pip install -r requirements.txt'."
            ) from exc

        return easyocr.Reader(["en"], gpu=False, verbose=False)

    @classmethod
    def recognize_image(cls, image_bytes: bytes) -> Dict[str, Any]:
        """Recognize text in image bytes and return normalized OCR data."""
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                rgb_image = np.asarray(image.convert("RGB"))
            raw_results = cls._get_reader().readtext(
                rgb_image,
                detail=1,
                paragraph=False
            )
        except LocalOCRError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise LocalOCRError(f"EasyOCR could not process this image: {exc}") from exc

        return format_easyocr_results(raw_results)
