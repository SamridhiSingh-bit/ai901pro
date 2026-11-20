"""
services/vision_service.py
==============================================================================
Azure AI Vision Integration Service
------------------------------------------------------------------------------
Implements secure connection handling, validation, and multi-modal image analysis
using Microsoft Azure AI Vision Image Analysis 4.0 (Python SDK & REST API).

Aligned with Microsoft Azure AI Fundamentals (AI-900 / AI-901) core workloads:
- Computer Vision (Image Classification & Tagging)
- Object Detection (Bounding boxes and localized coordinates)
- Image Captioning & Dense Captioning (Scene semantics)
- Optical Character Recognition / OCR (Read API for text in images)
==============================================================================
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision_service")

# Modern Azure AI Vision SDK imports
try:
    from azure.ai.vision.imageanalysis import ImageAnalysisClient
    from azure.ai.vision.imageanalysis.models import VisualFeatures
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import (
        ClientAuthenticationError,
        ResourceNotFoundError,
        HttpResponseError,
        ServiceRequestError
    )
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False
    logger.warning("azure-ai-vision-imageanalysis SDK not found. Falling back to direct REST API.")


class AzureVisionService:
    """
    Manages communication with Azure AI Vision 4.0 service.
    Supports credential validation, connectivity diagnostics, live multi-modal
    analysis, and explicit demo data when offline.
    """

    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize the vision service. Loads credentials from environment if not provided.
        """
        # Reload environment variables to pick up any changes in .env
        load_dotenv(override=False)

        raw_endpoint = endpoint or os.getenv("AZURE_VISION_ENDPOINT", "")
        raw_key = key or os.getenv("AZURE_VISION_KEY", "")

        self.endpoint = self._sanitize_endpoint(raw_endpoint)
        self.key = raw_key.strip()
        self.api_version = "2024-02-01"

    @staticmethod
    def _sanitize_endpoint(endpoint: str) -> str:
        """Sanitizes and normalizes the Azure endpoint URL."""
        if not endpoint:
            return ""
        ep = endpoint.strip()
        # Add https if missing
        if not ep.startswith("http://") and not ep.startswith("https://"):
            ep = f"https://{ep}"
        # Ensure trailing slash for consistent SDK handling
        if not ep.endswith("/"):
            ep = f"{ep}/"
        return ep

    def get_masked_key(self) -> str:
        """Returns a masked version of the key for safe UI display."""
        if not self.key or self.is_placeholder():
            return "Not Configured"
        if len(self.key) <= 8:
            return "••••••••"
        return f"{self.key[:4]}••••••••{self.key[-4:]}"

    def is_configured(self) -> bool:
        """Checks whether the service has non-empty, non-placeholder credentials."""
        if not self.endpoint or not self.key:
            return False
        if self.is_placeholder():
            return False
        return True

    def is_placeholder(self) -> bool:
        """Checks if configured values are sample placeholders."""
        placeholders = [
            "your-resource-name",
            "your_azure_vision_key_here",
            "your_endpoint_here",
            "example.com",
            "<your-resource-name>"
        ]
        return any(p in self.endpoint.lower() or p in self.key.lower() for p in placeholders)

    def validate_configuration(self) -> Tuple[bool, str]:
        """
        Validates the configuration structure prior to network calls.
        Returns (is_valid, error_message).
        """
        if not self.endpoint:
            return False, "Azure Vision Endpoint is missing. Please set AZURE_VISION_ENDPOINT in .env."

        parsed = urlparse(self.endpoint)
        if not parsed.scheme or not parsed.netloc:
            return False, f"Invalid Azure Vision Endpoint format: '{self.endpoint}'. Must be a valid URL."

        if not self.key:
            return False, "Azure Vision Key is missing. Please set AZURE_VISION_KEY in .env."

        if self.is_placeholder():
            return False, "Placeholder credentials detected. Please replace placeholder values with your real Azure keys."

        if len(self.key) < 16:
            return False, "Azure Vision Key appears too short (expected 32 hex characters)."

        return True, "Configuration format is valid."

    def test_connectivity(self) -> Dict[str, Any]:
        """
        Runs a lightweight diagnostic test against the Azure AI Vision endpoint.
        Does not pretend to succeed: returns exact HTTP status and diagnostic message.
        """
        is_valid, msg = self.validate_configuration()
        if not is_valid:
            return {
                "status": "error",
                "code": "CONFIG_INVALID",
                "message": msg,
                "endpoint": self.endpoint,
                "masked_key": self.get_masked_key()
            }

        # Azure Image Analysis 4.0 analysis endpoint
        test_url = f"{self.endpoint.rstrip('/')}/computervision/imageanalysis:analyze"
        params = {
            "api-version": self.api_version,
            "features": "tags"
        }
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json"
        }

        # Minimal test payload (1x1 transparent PNG as data URL or public test URL)
        body = {
            "url": "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"
        }

        try:
            resp = requests.post(test_url, params=params, headers=headers, json=body, timeout=8)
            if resp.status_code == 200:
                return {
                    "status": "success",
                    "code": "OK_200",
                    "message": "Azure AI Vision connection verified successfully! API authenticated and reachable.",
                    "endpoint": self.endpoint,
                    "masked_key": self.get_masked_key(),
                    "api_version": self.api_version
                }
            elif resp.status_code == 401:
                return {
                    "status": "error",
                    "code": "AUTH_401",
                    "message": "Authentication failed (HTTP 401 Unauthorized). The provided Azure Vision Key is invalid or expired.",
                    "endpoint": self.endpoint,
                    "masked_key": self.get_masked_key()
                }
            elif resp.status_code == 404:
                return {
                    "status": "error",
                    "code": "NOT_FOUND_404",
                    "message": f"Resource not found (HTTP 404). Endpoint '{self.endpoint}' may be misspelled or does not support Vision 4.0.",
                    "endpoint": self.endpoint,
                    "masked_key": self.get_masked_key()
                }
            elif resp.status_code == 429:
                return {
                    "status": "error",
                    "code": "RATE_LIMIT_429",
                    "message": "Rate limit exceeded (HTTP 429). Resource quota or call limit reached.",
                    "endpoint": self.endpoint,
                    "masked_key": self.get_masked_key()
                }
            else:
                error_details = resp.text
                return {
                    "status": "error",
                    "code": f"HTTP_{resp.status_code}",
                    "message": f"Azure returned HTTP {resp.status_code}: {error_details}",
                    "endpoint": self.endpoint,
                    "masked_key": self.get_masked_key()
                }
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "code": "TIMEOUT",
                "message": "Connection to Azure Vision endpoint timed out after 8 seconds. Check internet connectivity and endpoint URL.",
                "endpoint": self.endpoint,
                "masked_key": self.get_masked_key()
            }
        except requests.exceptions.ConnectionError as e:
            return {
                "status": "error",
                "code": "DNS_CONNECTION_ERROR",
                "message": f"Could not reach Azure endpoint. Network or DNS resolution error: {str(e)}",
                "endpoint": self.endpoint,
                "masked_key": self.get_masked_key()
            }
        except Exception as e:
            return {
                "status": "error",
                "code": "UNEXPECTED_ERROR",
                "message": f"Diagnostic check encountered an unexpected error: {str(e)}",
                "endpoint": self.endpoint,
                "masked_key": self.get_masked_key()
            }

    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Performs full multi-modal analysis on the uploaded image binary.
        Extracts: Caption, Dense Captions, Tags, Objects (with bounding boxes), and OCR (Read).

        Returns a structured dictionary or raises an explicit exception.
        """
        is_valid, msg = self.validate_configuration()
        if not is_valid:
            return {
                "success": False,
                "error_type": "CONFIGURATION_ERROR",
                "message": msg,
                "is_demo": False
            }

        # Attempt with Azure SDK first if available
        if AZURE_SDK_AVAILABLE:
            try:
                return self._analyze_with_sdk(image_bytes)
            except (ClientAuthenticationError, HttpResponseError, ServiceRequestError) as sdk_err:
                logger.warning(f"Azure SDK call failed: {sdk_err}. Retrying via direct REST API.")
                # Fallback to direct REST API
                return self._analyze_with_rest(image_bytes)
            except Exception as e:
                logger.error(f"Unexpected error in SDK analysis: {e}")
                return self._analyze_with_rest(image_bytes)
        else:
            return self._analyze_with_rest(image_bytes)

    def _analyze_with_sdk(self, image_bytes: bytes) -> Dict[str, Any]:
        """Executes analysis using azure-ai-vision-imageanalysis SDK."""
        client = ImageAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )

        visual_features = [
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.READ,
            VisualFeatures.PEOPLE,
            VisualFeatures.SMART_CROPS
        ]

        result = client.analyze(
            image_data=image_bytes,
            visual_features=visual_features
        )

        # Parse SDK result into clean JSON-serializable structure
        raw_dict = result.as_dict() if hasattr(result, "as_dict") else {}

        return self._format_analysis_result(raw_dict, source="azure_sdk_4.0")

    def _analyze_with_rest(self, image_bytes: bytes) -> Dict[str, Any]:
        """Executes analysis using direct Azure AI Vision 4.0 REST API."""
        url = f"{self.endpoint.rstrip('/')}/computervision/imageanalysis:analyze"
        params = {
            "api-version": self.api_version,
            "features": "caption,denseCaptions,tags,objects,read,people,smartCrops"
        }
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/octet-stream"
        }

        try:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                data=image_bytes,
                timeout=25
            )

            if response.status_code == 200:
                raw_dict = response.json()
                return self._format_analysis_result(raw_dict, source="azure_rest_4.0")
            elif response.status_code == 401:
                return {
                    "success": False,
                    "error_type": "AUTHENTICATION_FAILED",
                    "message": "Azure rejected credentials (HTTP 401 Unauthorized). Verify your AZURE_VISION_KEY in .env.",
                    "status_code": 401
                }
            elif response.status_code == 404:
                return {
                    "success": False,
                    "error_type": "RESOURCE_NOT_FOUND",
                    "message": f"Azure endpoint returned HTTP 404. Verify AZURE_VISION_ENDPOINT: {self.endpoint}",
                    "status_code": 404
                }
            elif response.status_code == 429:
                return {
                    "success": False,
                    "error_type": "RATE_LIMITED",
                    "message": "Azure AI Vision rate limit reached (HTTP 429). Please wait before trying again.",
                    "status_code": 429
                }
            else:
                return {
                    "success": False,
                    "error_type": f"HTTP_ERROR_{response.status_code}",
                    "message": f"Azure API error (HTTP {response.status_code}): {response.text}",
                    "status_code": response.status_code
                }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error_type": "TIMEOUT",
                "message": "Request to Azure Vision service timed out. Check your network connection."
            }
        except requests.exceptions.ConnectionError as ce:
            return {
                "success": False,
                "error_type": "CONNECTION_ERROR",
                "message": f"Failed to connect to Azure Vision endpoint: {str(ce)}"
            }
        except Exception as ex:
            return {
                "success": False,
                "error_type": "UNEXPECTED_ERROR",
                "message": f"Unexpected error while calling Azure API: {str(ex)}"
            }

    def _format_analysis_result(self, raw: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Normalizes Azure AI Vision 4.0 response into a clean, comprehensive dictionary.
        """
        # Metadata
        metadata = raw.get("metadata", {})
        model_version = raw.get("modelVersion", "2024-02-01")

        # 1. Caption
        caption_data = raw.get("captionResult", {})
        caption = {
            "text": caption_data.get("text", "No caption generated."),
            "confidence": float(caption_data.get("confidence", 0.0)),
            "confidence_pct": round(float(caption_data.get("confidence", 0.0)) * 100, 2)
        }

        # 2. Dense Captions
        dense_captions = []
        for dc in raw.get("denseCaptionsResult", {}).get("values", []):
            bb = dc.get("boundingBox", {})
            dense_captions.append({
                "text": dc.get("text", ""),
                "confidence": float(dc.get("confidence", 0.0)),
                "confidence_pct": round(float(dc.get("confidence", 0.0)) * 100, 2),
                "bounding_box": {
                    "x": bb.get("x", 0),
                    "y": bb.get("y", 0),
                    "width": bb.get("w", bb.get("width", 0)),
                    "height": bb.get("h", bb.get("height", 0))
                }
            })

        # 3. Tags
        tags = []
        for t in raw.get("tagsResult", {}).get("values", []):
            conf = float(t.get("confidence", 0.0))
            tags.append({
                "name": t.get("name", ""),
                "confidence": conf,
                "confidence_pct": round(conf * 100, 2)
            })

        # 4. Objects (with bounding box)
        objects = []
        for obj in raw.get("objectsResult", {}).get("values", []):
            bb = obj.get("boundingBox", {})
            # Tag list in detected object
            obj_tags = obj.get("tags", [])
            primary_name = "Object"
            primary_conf = 0.0
            if obj_tags:
                primary_name = obj_tags[0].get("name", "Object")
                primary_conf = float(obj_tags[0].get("confidence", 0.0))

            objects.append({
                "name": primary_name,
                "confidence": primary_conf,
                "confidence_pct": round(primary_conf * 100, 2),
                "bounding_box": {
                    "x": bb.get("x", 0),
                    "y": bb.get("y", 0),
                    "width": bb.get("w", bb.get("width", 0)),
                    "height": bb.get("h", bb.get("height", 0))
                }
            })

        # 5. Read / OCR
        read_data = raw.get("readResult", {})
        blocks = read_data.get("blocks", [])
        lines = []
        full_text_lines = []
        total_words = 0

        for b in blocks:
            for line in b.get("lines", []):
                line_text = line.get("text", "").strip()
                if line_text:
                    full_text_lines.append(line_text)
                    words = line.get("words", [])
                    total_words += len(words)
                    polygon = line.get("boundingPolygon", [])
                    lines.append({
                        "text": line_text,
                        "words_count": len(words),
                        "bounding_polygon": polygon
                    })

        full_text = "\n".join(full_text_lines)

        # 6. People detection
        people = []
        for p in raw.get("peopleResult", {}).get("values", []):
            bb = p.get("boundingBox", {})
            people.append({
                "confidence": float(p.get("confidence", 0.0)),
                "confidence_pct": round(float(p.get("confidence", 0.0)) * 100, 2),
                "bounding_box": {
                    "x": bb.get("x", 0),
                    "y": bb.get("y", 0),
                    "width": bb.get("w", bb.get("width", 0)),
                    "height": bb.get("h", bb.get("height", 0))
                }
            })

        return {
            "success": True,
            "source": source,
            "is_demo": False,
            "model_version": model_version,
            "metadata": {
                "width": metadata.get("width", 0),
                "height": metadata.get("height", 0)
            },
            "caption": caption,
            "dense_captions": dense_captions,
            "tags": tags,
            "objects": objects,
            "people": people,
            "ocr": {
                "detected": len(lines) > 0,
                "total_lines": len(lines),
                "total_words": total_words,
                "lines": lines,
                "full_text": full_text
            },
            "raw_json": raw
        }

    @staticmethod
    def get_sample_demo_analysis(preset_key: str) -> Dict[str, Any]:
        """
        Returns structured, realistic sample analysis for UI review & demonstration.
        EXPLICITLY FLAGGED as demo/offline so users and faculty know Azure was not called.
        """
        samples = {
            "street": {
                "title": "Urban Traffic Scene",
                "caption": {
                    "text": "A busy city street with multiple vehicles, a pedestrian crosswalk, and traffic signals",
                    "confidence": 0.942,
                    "confidence_pct": 94.20
                },
                "tags": [
                    {"name": "vehicle", "confidence": 0.985, "confidence_pct": 98.50},
                    {"name": "car", "confidence": 0.971, "confidence_pct": 97.10},
                    {"name": "street", "confidence": 0.963, "confidence_pct": 96.30},
                    {"name": "road", "confidence": 0.954, "confidence_pct": 95.40},
                    {"name": "traffic light", "confidence": 0.928, "confidence_pct": 92.80},
                    {"name": "pedestrian", "confidence": 0.891, "confidence_pct": 89.10},
                    {"name": "building", "confidence": 0.874, "confidence_pct": 87.40},
                    {"name": "urban area", "confidence": 0.862, "confidence_pct": 86.20}
                ],
                "objects": [
                    {"name": "car", "confidence": 0.94, "confidence_pct": 94.0, "bounding_box": {"x": 50, "y": 280, "width": 260, "height": 180}},
                    {"name": "car", "confidence": 0.91, "confidence_pct": 91.0, "bounding_box": {"x": 340, "y": 290, "width": 240, "height": 170}},
                    {"name": "person", "confidence": 0.88, "confidence_pct": 88.0, "bounding_box": {"x": 620, "y": 240, "width": 90, "height": 220}},
                    {"name": "traffic light", "confidence": 0.85, "confidence_pct": 85.0, "bounding_box": {"x": 180, "y": 60, "width": 60, "height": 110}}
                ],
                "dense_captions": [
                    {"text": "a silver car driving on the asphalt road", "confidence": 0.92, "confidence_pct": 92.0, "bounding_box": {"x": 50, "y": 280, "width": 260, "height": 180}},
                    {"text": "a pedestrian waiting at the crosswalk", "confidence": 0.88, "confidence_pct": 88.0, "bounding_box": {"x": 620, "y": 240, "width": 90, "height": 220}},
                    {"text": "an overhead street traffic light showing green", "confidence": 0.84, "confidence_pct": 84.0, "bounding_box": {"x": 180, "y": 60, "width": 60, "height": 110}}
                ],
                "people": [
                    {"confidence": 0.88, "confidence_pct": 88.0, "bounding_box": {"x": 620, "y": 240, "width": 90, "height": 220}}
                ],
                "ocr": {
                    "detected": True,
                    "total_lines": 3,
                    "total_words": 7,
                    "lines": [
                        {"text": "MAIN ST", "words_count": 2, "bounding_polygon": [{"x": 170, "y": 180}, {"x": 250, "y": 180}, {"x": 250, "y": 205}, {"x": 170, "y": 205}]},
                        {"text": "SPEED LIMIT 25", "words_count": 3, "bounding_polygon": [{"x": 680, "y": 140}, {"x": 780, "y": 140}, {"x": 780, "y": 190}, {"x": 680, "y": 190}]},
                        {"text": "ONE WAY", "words_count": 2, "bounding_polygon": [{"x": 680, "y": 200}, {"x": 770, "y": 200}, {"x": 770, "y": 230}, {"x": 680, "y": 230}]}
                    ],
                    "full_text": "MAIN ST\nSPEED LIMIT 25\nONE WAY"
                },
                "metadata": {"width": 800, "height": 550}
            },
            "presentation": {
                "title": "Technical Conference Slide",
                "caption": {
                    "text": "A technical presentation slide displaying cloud AI architecture diagrams and text bullet points",
                    "confidence": 0.965,
                    "confidence_pct": 96.50
                },
                "tags": [
                    {"name": "text", "confidence": 0.992, "confidence_pct": 99.20},
                    {"name": "screenshot", "confidence": 0.981, "confidence_pct": 98.10},
                    {"name": "diagram", "confidence": 0.945, "confidence_pct": 94.50},
                    {"name": "software", "confidence": 0.912, "confidence_pct": 91.20},
                    {"name": "presentation", "confidence": 0.895, "confidence_pct": 89.50},
                    {"name": "font", "confidence": 0.880, "confidence_pct": 88.00}
                ],
                "objects": [
                    {"name": "screen", "confidence": 0.92, "confidence_pct": 92.0, "bounding_box": {"x": 40, "y": 30, "width": 720, "height": 480}},
                    {"name": "laptop", "confidence": 0.86, "confidence_pct": 86.0, "bounding_box": {"x": 520, "y": 300, "width": 220, "height": 180}}
                ],
                "dense_captions": [
                    {"text": "slide title about artificial intelligence workloads", "confidence": 0.95, "confidence_pct": 95.0, "bounding_box": {"x": 60, "y": 50, "width": 680, "height": 70}},
                    {"text": "bullet points explaining computer vision fundamentals", "confidence": 0.91, "confidence_pct": 91.0, "bounding_box": {"x": 60, "y": 140, "width": 450, "height": 280}}
                ],
                "people": [],
                "ocr": {
                    "detected": True,
                    "total_lines": 6,
                    "total_words": 28,
                    "lines": [
                        {"text": "Microsoft Azure AI Fundamentals (AI-900)", "words_count": 5, "bounding_polygon": []},
                        {"text": "Computer Vision Workloads & Core Capabilities:", "words_count": 6, "bounding_polygon": []},
                        {"text": "1. Image Classification & Semantic Tagging", "words_count": 6, "bounding_polygon": []},
                        {"text": "2. Object Detection with Bounding Box Coordinates", "words_count": 7, "bounding_polygon": []},
                        {"text": "3. Optical Character Recognition (OCR Read API)", "words_count": 7, "bounding_polygon": []},
                        {"text": "4. Natural Language Image Caption Generation", "words_count": 6, "bounding_polygon": []}
                    ],
                    "full_text": "Microsoft Azure AI Fundamentals (AI-900)\nComputer Vision Workloads & Core Capabilities:\n1. Image Classification & Semantic Tagging\n2. Object Detection with Bounding Box Coordinates\n3. Optical Character Recognition (OCR Read API)\n4. Natural Language Image Caption Generation"
                },
                "metadata": {"width": 800, "height": 550}
            },
            "document": {
                "title": "Business Invoice Document",
                "caption": {
                    "text": "A printed commercial invoice document with itemized billing table and company contact details",
                    "confidence": 0.978,
                    "confidence_pct": 97.80
                },
                "tags": [
                    {"name": "invoice", "confidence": 0.990, "confidence_pct": 99.00},
                    {"name": "receipt", "confidence": 0.975, "confidence_pct": 97.50},
                    {"name": "text", "confidence": 0.965, "confidence_pct": 96.50},
                    {"name": "document", "confidence": 0.952, "confidence_pct": 95.20},
                    {"name": "number", "confidence": 0.910, "confidence_pct": 91.00}
                ],
                "objects": [
                    {"name": "document", "confidence": 0.95, "confidence_pct": 95.0, "bounding_box": {"x": 50, "y": 30, "width": 700, "height": 490}}
                ],
                "dense_captions": [
                    {"text": "header with invoice number and billing date", "confidence": 0.94, "confidence_pct": 94.0, "bounding_box": {"x": 60, "y": 40, "width": 680, "height": 80}},
                    {"text": "itemized financial charges and grand total", "confidence": 0.92, "confidence_pct": 92.0, "bounding_box": {"x": 60, "y": 200, "width": 680, "height": 220}}
                ],
                "people": [],
                "ocr": {
                    "detected": True,
                    "total_lines": 7,
                    "total_words": 26,
                    "lines": [
                        {"text": "INVOICE #INV-2026-0891", "words_count": 2, "bounding_polygon": []},
                        {"text": "Billed To: Cloud Tech Solutions Pvt Ltd", "words_count": 7, "bounding_polygon": []},
                        {"text": "Date of Issue: November 20, 2026", "words_count": 5, "bounding_polygon": []},
                        {"text": "Item 1: Azure AI Vision Cognitive API - $120.00", "words_count": 8, "bounding_polygon": []},
                        {"text": "Item 2: Cloud Storage Hot Tier - $35.00", "words_count": 7, "bounding_polygon": []},
                        {"text": "Subtotal: $155.00 | Tax (10%): $15.50", "words_count": 6, "bounding_polygon": []},
                        {"text": "Total Due: $170.50 USD", "words_count": 4, "bounding_polygon": []}
                    ],
                    "full_text": "INVOICE #INV-2026-0891\nBilled To: Cloud Tech Solutions Pvt Ltd\nDate of Issue: November 20, 2026\nItem 1: Azure AI Vision Cognitive API - $120.00\nItem 2: Cloud Storage Hot Tier - $35.00\nSubtotal: $155.00 | Tax (10%): $15.50\nTotal Due: $170.50 USD"
                },
                "metadata": {"width": 800, "height": 550}
            }
        }

        data = samples.get(preset_key, samples["street"])

        return {
            "success": True,
            "source": "demo_offline_sample",
            "is_demo": True,
            "demo_preset": preset_key,
            "model_version": "2024-02-01-demo-preset",
            "metadata": data["metadata"],
            "caption": data["caption"],
            "dense_captions": data["dense_captions"],
            "tags": data["tags"],
            "objects": data["objects"],
            "people": data["people"],
            "ocr": data["ocr"],
            "raw_json": {
                "_disclaimer": "OFFLINE DEMO MODE: Pre-packaged sample result for UI testing without live Azure subscription.",
                "preset": preset_key,
                "captionResult": data["caption"],
                "tagsResult": {"values": data["tags"]},
                "objectsResult": {"values": data["objects"]},
                "readResult": {"lines": data["ocr"]["lines"], "fullText": data["ocr"]["full_text"]}
            }
        }
