# AI-Based Image Recognition and Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![Azure AI](https://img.shields.io/badge/Microsoft%20Azure-AI%20Vision%204.0-0078D4.svg)](https://azure.microsoft.com/en-us/products/ai-services/ai-vision)
[![License](https://img.shields.io/badge/Academic%20Project-BE%20CSE%20(AI%26ML)-green.svg)]()

An enterprise-grade multi-modal visual intelligence platform engineered using **Python**, **Streamlit**, and **Microsoft Azure AI Vision 4.0** (backed by the Microsoft Florence vision foundation model).

Designed as a college-level academic project aligned with **Microsoft Azure AI Fundamentals (AI-900 / AI-901)** concepts.

---

## 🌟 Key Features

### 1. Unified Multi-Modal AI Vision
- **Natural Language Captioning:** Generates comprehensive, human-readable scene descriptions using the Microsoft Florence foundation model.
- **Dense Sub-Region Captioning:** Discovers and describes multiple distinct regions within a single image.
- **Multi-Class Object Detection:** Identifies entities, outputs spatial coordinates $(X, Y, W, H)$, and draws high-contrast bounding boxes with confidence labels using **Pillow**.
- **Semantic Concept Tagging:** Classifies images across thousands of visual categories with statistical confidence progress bars.
- **Optical Character Recognition (OCR):** Azure Read API for live Azure analysis, plus optional local English OCR for uploaded images via EasyOCR (CPU; no Azure key required). Local results include recognized lines, confidence scores, highlighted text regions, and a downloadable text file.

### 2. Modern Dark-Themed UI/UX
- Near-black canvas (`#0A0E17`) with charcoal cards (`#141B2D`) and Azure blue (`#0078D4` / `#38BDF8`) accents.
- Responsive multi-tab organization: *Overview*, *Object Detection*, *Image Tags*, *Image Captions*, *OCR Results*, and *Raw JSON & Downloads*.
- Interactive confidence threshold slider to dynamically filter detected objects and bounding box overlays.

### 3. Enterprise Security & Zero-Trust Architecture
- Cloud secrets stored strictly in a `.env` file excluded from version control via `.gitignore`.
- Automatic key masking (`••••••••`) in all UI displays and logs.
- Strict input validation: formats (JPG, JPEG, PNG), file size (up to 20 MB), and dimensions (50 px to 16,000 px).
- Comprehensive error handling for HTTP 401 (Unauthorized), HTTP 404 (Not Found), HTTP 429 (Rate Limits), and network faults.

### 4. Academic Resilience (Offline Demo Mode)
- **Zero-Cloud-Cost Demonstration:** Includes a dedicated offline showcase featuring pre-packaged academic scenarios (Traffic Scene, Conference Presentation, Business Invoice).
- Displays transparent academic demo banners—never pretends an analysis was called on Azure when running offline.
- The Live Image Analyzer also offers **Run Local OCR (EasyOCR)** for actual text recognition on uploaded images without Azure credentials. First use downloads EasyOCR's English model files; this requires an internet connection.

### 5. Multi-Format Data Export
- Download full analysis results matching official Azure Vision 4.0 schema as a **JSON** file.
- Download extracted OCR text as a formatted **TXT** file.
- Download annotated images with bounding boxes as high-resolution **PNG** files.

---

## 🏗️ Folder Structure

```
ImageRecognitionSystem/
│
├── app.py                     # Main Streamlit application entry point
├── requirements.txt           # Pinned production dependencies
├── .env.example               # Safe configuration template with instructions
├── .gitignore                 # Excludes secrets, bytecode, and virtual environments
├── README.md                  # Comprehensive project documentation
│
├── services/                  # Business & Cloud integration tier
│   ├── __init__.py
│   └── vision_service.py      # Azure AI Vision 4.0 client, REST fallback & diagnostics
│
├── utils/                     # Image processing and presentation tier
│   ├── __init__.py
│   ├── image_utils.py         # Pillow bounding box annotator, validation, asset synthesis
│   └── ui_components.py       # Custom dark CSS theme, metric cards, badges, banners
│
├── assets/                    # Curated sample images for offline demonstration
│   ├── sample_street.jpg      # Urban traffic preset (Vehicle & Pedestrian detection)
│   ├── sample_presentation.png# AI architecture slide preset (Dense OCR text)
│   └── sample_invoice.png     # Commercial invoice preset (Financial document OCR)
│
├── tests/                     # Automated testing suite
│   └── test_suite.py          # 11 unit & validation tests (imports, validation, security)
│
└── docs/                      # Academic university report documentation
    ├── project_overview.md    # Abstract, problem statement, objectives, tech stack
    ├── system_architecture.md # Sequence & component diagrams, JSON schemas
    ├── testing_documentation.md# Test cases, error matrix, verification report
    └── report_material.md     # Full project report text, viva pitch & top 10 Q&A
```

---

## ⚙️ Prerequisites

- **Python 3.10 to 3.13** installed on your system.
- An internet connection (for live Azure cloud analysis and initial package installation).
- *(Optional for Live Mode)* A Microsoft Azure account with an active **Azure AI Services** or **Computer Vision** resource.

---

## 🚀 Installation & Setup Instructions

### Step 1: Clone or Open the Project
Open your terminal (PowerShell or Command Prompt) and navigate to the project directory:

```bash
cd ImageRecognitionSystem
```

### Step 2: Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt - cmd):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## ☁️ Azure Cloud Configuration (For Live Analysis)

1. Sign in to the [Azure Portal](https://portal.azure.com/).
2. Click **Create a resource** and search for **Azure AI Services** (multi-service) or **Computer Vision** (single-service).
3. Select a region that supports Image Analysis 4.0 (e.g., `East US`, `West US 2`, `North Europe`).
4. Select the **Free F0** pricing tier (or Standard S0).
5. Once deployed, open your resource and navigate to **Resource Management** -> **Keys and Endpoint**.
6. Copy your **KEY 1** and **Endpoint URL**.
7. In the `ImageRecognitionSystem` directory, create a `.env` file by copying the example:

```bash
cp .env.example .env
```
*(On Windows PowerShell: `Copy-Item .env.example .env`)*

8. Open `.env` in any text editor and paste your credentials:
```env
AZURE_VISION_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/
AZURE_VISION_KEY=your_copied_key_1
```

> **Note:** If you do not have Azure credentials right now, you can still run the application immediately! It will launch in **Offline Academic Demo Mode**, enabling full evaluation of all UI tabs, bounding box drawing, and OCR layouts using pre-packaged sample data.

The offline showcase opens by default when Azure credentials are absent. Its captions, tags, OCR text, object boxes, and scores are pre-packaged examples for demonstrating the interface; they are not generated from uploaded images and do not represent live Azure analysis. Local image annotation and sample exports work without cloud credentials. To analyze arbitrary uploaded images with Azure, configure a valid endpoint and key.

---

## 🏃 How to Run the Application

Execute the following command in your terminal:

```bash
streamlit run app.py
```

Your browser will automatically open the dashboard at:
👉 **`http://localhost:8501`**

---

## 🧪 Running Automated Reliability Tests

Before presenting to faculty, execute the automated verification test suite:

```bash
python tests/test_suite.py
```

**Expected Output:**
```text
...........
----------------------------------------------------------------------
Ran 11 tests in 4.198s

OK
```

All 11 tests validate imports, file validation, Pillow bounding box drawing, missing-credential safeguards, demo presets, and secret scanning.

---

## 🛠️ Troubleshooting Guide

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'streamlit'` | Virtual environment not activated or dependencies not installed. | Activate virtual environment and run `pip install -r requirements.txt`. |
| `Authentication failed (HTTP 401 Unauthorized)` | Invalid or expired Azure key. | Check `AZURE_VISION_KEY` in `.env`. Ensure there are no leading/trailing spaces. |
| `Resource not found (HTTP 404)` | Endpoint URL is misspelled or wrong resource type. | Ensure `AZURE_VISION_ENDPOINT` matches `https://<name>.cognitiveservices.azure.com/`. |
| `Rate limit exceeded (HTTP 429)` | Exceeded the 20 calls/minute limit of Azure Free F0 tier. | Wait 60 seconds before submitting the next image. |
| `Cannot perform live analysis: Azure Vision credentials are not configured` | `.env` file is missing or contains placeholder values. | Enter valid keys in `.env` or switch to the **Offline Demo Showcase** in the sidebar. |

---

## 📸 Screenshots & Demonstration Artifacts

*(Capture these screenshots from your running application to include in your university report)*

- **Dashboard & System Overview:** `[SCREENSHOT: Main dashboard with dark theme and capability cards]`
- **Image Upload & Verification:** `[SCREENSHOT: File metadata card showing format, size, dimensions]`
- **Object Detection & Bounding Boxes:** `[SCREENSHOT: Pillow bounding boxes and detected entities table]`
- **Semantic Tagging:** `[SCREENSHOT: Confidence progress bars and concept cloud]`
- **OCR Text Extraction:** `[SCREENSHOT: Extracted lines and copyable text area]`
- **Azure Diagnostics:** `[SCREENSHOT: Live connectivity health check dialog]`
- **Offline Demo Mode:** `[SCREENSHOT: Curated scenario running with academic disclaimer banner]`

---

## 🎓 Academic Viva Voce Tips
- **Key Concept:** Explain how Microsoft Azure AI Vision 4.0 uses the unified **Florence foundation model** instead of multiple disjoint CNN models.
- **Security:** Highlight how secrets are isolated in `.env` and masked in the UI.
- **Resilience:** Mention the **Offline Academic Demo Mode** that allows faculty evaluation even without cloud credits.
- Refer to `docs/report_material.md` for the complete 3-minute pitch and top 10 viva questions with detailed answers!
