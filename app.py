"""
app.py
==============================================================================
AI-Based Image Recognition and Analysis System
------------------------------------------------------------------------------
A production-grade multi-modal image intelligence dashboard powered by
Microsoft Azure AI Vision 4.0 and Streamlit.

Aligned with Microsoft Azure AI Fundamentals (AI-900 / AI-901).
==============================================================================
"""

import os
import json
import io
from PIL import Image
import pandas as pd
import streamlit as st

# Local modules
from services.vision_service import AzureVisionService
from utils.image_utils import (
    validate_image_file,
    annotate_objects_on_image,
    annotate_ocr_lines_on_image,
    image_to_bytes,
    generate_sample_assets_if_missing
)
from utils.ui_components import (
    inject_custom_theme,
    render_header,
    render_disclaimer_banner,
    render_caption_card,
    render_metric_card,
    render_confidence_bar_html,
    render_footer
)

# ------------------------------------------------------------------------------
# 1. Page Configuration & Setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Azure AI Vision | Image Recognition & Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom modern dark theme
inject_custom_theme()

# Ensure demo assets exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
generate_sample_assets_if_missing(ASSETS_DIR)

# Initialize Session State
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if "uploaded_metadata" not in st.session_state:
    st.session_state.uploaded_metadata = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "active_preset" not in st.session_state:
    st.session_state.active_preset = None

# Initialize Vision Service
vision_service = AzureVisionService()
is_azure_configured = vision_service.is_configured()

# ------------------------------------------------------------------------------
# 2. Sidebar Navigation & Diagnostics
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.8rem;">👁️</span>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #ffffff;">Azure AI Vision</div>
                <div style="font-size: 0.72rem; color: #38bdf8; letter-spacing: 0.05em; text-transform: uppercase;">Analysis System 4.0</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Main Navigation
    st.markdown("### 📌 Navigation")
    app_mode = st.radio(
        label="Select Application Workspace",
        options=[
            "🚀 Live Image Analyzer",
            "🏛️ Offline Demo Showcase",
            "📖 Architecture & AI-900 Concepts",
            "⚙️ Azure Configuration & Diagnostics"
        ],
        index=0 if is_azure_configured else 1,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Cloud Connection Status Card
    st.markdown("### ☁️ Cloud Service Status")
    if is_azure_configured:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 8px; padding: 0.8rem; margin-bottom: 0.8rem;">
                <div style="color: #34d399; font-weight: 700; font-size: 0.85rem; display: flex; align-items: center; gap: 0.4rem;">
                    <span>●</span> Azure Vision Configured
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.3rem;">
                    Endpoint: <code style="color: #e2e8f0;">{vision_service.endpoint[:28]}...</code><br>
                    Key: <code style="color: #e2e8f0;">{vision_service.get_masked_key()}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 8px; padding: 0.8rem; margin-bottom: 0.8rem;">
                <div style="color: #fbbf24; font-weight: 700; font-size: 0.85rem; display: flex; align-items: center; gap: 0.4rem;">
                    <span>●</span> Credentials Missing
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.3rem;">
                    Azure credentials are not set in <code>.env</code>. You can test live images once configured or use the <strong>Demo Showcase</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Quick Connectivity Diagnostic Trigger
    if st.button("🔍 Test Azure Connectivity", use_container_width=True):
        with st.spinner("Testing Azure Vision API connection..."):
            diag = vision_service.test_connectivity()
            if diag["status"] == "success":
                st.success(f"Connected! {diag['message']}")
            else:
                st.error(f"Connection Failed: {diag['message']}")

    st.markdown("---")

    # Visualization Controls
    st.markdown("### 🎛️ Visualization Settings")
    min_conf_slider = st.slider(
        "Min Object Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="Filter detected objects and bounding boxes below this confidence score."
    )
    show_box_labels = st.checkbox("Show Labels on Bounding Boxes", value=True)
    show_box_conf = st.checkbox("Show Confidence on Bounding Boxes", value=True)
    box_thickness = st.slider("Bounding Box Thickness", min_value=1, max_value=6, value=3)

    st.markdown("---")
    st.caption("BE CSE (AI & ML) Project • AI-900 Academic Demonstration")


# ------------------------------------------------------------------------------
# WORKSPACE 1: 🚀 LIVE IMAGE ANALYZER
# ------------------------------------------------------------------------------
if app_mode == "🚀 Live Image Analyzer":
    render_header(
        title="Live Image Recognition & Analysis",
        subtitle="Extract captions, identify multi-class objects with bounding boxes, discover tags, and perform OCR using Azure AI Vision 4.0.",
        is_live=is_azure_configured
    )

    # Quick overview of system capabilities
    st.markdown(
        """
        <div class="ai-card" style="margin-bottom: 1.5rem;">
            <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.4rem; color: #ffffff;">
                System Overview & AI Capabilities
            </div>
            <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6;">
                This system connects to <strong>Microsoft Azure AI Vision</strong> (Image Analysis 4.0), utilizing Microsoft's foundation vision model
                to perform unified image understanding. The service analyzes image features in a single API pass:
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.8rem; margin-top: 0.9rem;">
                <div style="background: #111a2a; padding: 0.7rem 0.9rem; border-radius: 6px; border: 1px solid #1f2d45;">
                    <div style="color: #38bdf8; font-weight: 700; font-size: 0.84rem;">1. Image Captioning</div>
                    <div style="color: #94a3b8; font-size: 0.76rem;">Natural language sentence describing scene semantics.</div>
                </div>
                <div style="background: #111a2a; padding: 0.7rem 0.9rem; border-radius: 6px; border: 1px solid #1f2d45;">
                    <div style="color: #10b981; font-weight: 700; font-size: 0.84rem;">2. Object Detection</div>
                    <div style="color: #94a3b8; font-size: 0.76rem;">Localized coordinates (X, Y, W, H) and bounding boxes.</div>
                </div>
                <div style="background: #111a2a; padding: 0.7rem 0.9rem; border-radius: 6px; border: 1px solid #1f2d45;">
                    <div style="color: #f59e0b; font-weight: 700; font-size: 0.84rem;">3. Image Tagging</div>
                    <div style="color: #94a3b8; font-size: 0.76rem;">Taxonomic tags for thousands of visual concepts.</div>
                </div>
                <div style="background: #111a2a; padding: 0.7rem 0.9rem; border-radius: 6px; border: 1px solid #1f2d45;">
                    <div style="color: #a855f7; font-weight: 700; font-size: 0.84rem;">4. Optical Character Recognition</div>
                    <div style="color: #94a3b8; font-size: 0.76rem;">Line and word-level text extraction (Read API).</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Image Upload Section
    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📤 Image Upload")
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, JPEG, PNG • Max 20 MB):",
            type=["jpg", "jpeg", "png"],
            key="user_image_uploader",
            help="Select a clear image containing objects, scenes, or text to analyze."
        )

        if uploaded_file is not None:
            # Validate uploaded file
            is_valid, err_msg, meta = validate_image_file(uploaded_file, uploaded_file.name)

            if is_valid:
                st.session_state.uploaded_image_bytes = bytes(uploaded_file.getbuffer())
                st.session_state.uploaded_metadata = meta

                # Display file metadata
                st.markdown(
                    f"""
                    <div class="ai-card" style="margin-top: 1rem;">
                        <div style="font-weight: 700; font-size: 0.92rem; color: #38bdf8; margin-bottom: 0.6rem;">
                            📋 Image Metadata & Verification
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; font-size: 0.82rem;">
                            <div><span style="color: #64748b;">Filename:</span> <strong style="color: #f1f5f9;">{meta['filename']}</strong></div>
                            <div><span style="color: #64748b;">Format:</span> <strong style="color: #f1f5f9;">{meta['format']}</strong></div>
                            <div><span style="color: #64748b;">Dimensions:</span> <strong style="color: #f1f5f9;">{meta['dimensions']}</strong></div>
                            <div><span style="color: #64748b;">File Size:</span> <strong style="color: #f1f5f9;">{meta['file_size_readable']}</strong></div>
                            <div><span style="color: #64748b;">Aspect Ratio:</span> <strong style="color: #f1f5f9;">{meta['aspect_ratio']}</strong></div>
                            <div><span style="color: #64748b;">Color Mode:</span> <strong style="color: #f1f5f9;">{meta['mode']}</strong></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Reset / Clear option
                if st.button("🗑️ Remove / Reset Uploaded Image", key="clear_img_btn"):
                    st.session_state.uploaded_image_bytes = None
                    st.session_state.uploaded_metadata = None
                    st.session_state.analysis_results = None
                    st.rerun()

            else:
                st.error(f"Validation Error: {err_msg}")
                st.session_state.uploaded_image_bytes = None

    with col_preview:
        st.markdown("### 🖼️ Image Preview")
        if st.session_state.uploaded_image_bytes:
            image_preview = Image.open(io.BytesIO(st.session_state.uploaded_image_bytes))
            st.image(image_preview, caption=f"Uploaded: {st.session_state.uploaded_metadata['filename']}", use_container_width=True)
        else:
            st.markdown(
                """
                <div style="border: 2px dashed #243550; border-radius: 12px; height: 260px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #64748b; background-color: #0b111c;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🖼️</div>
                    <div>No image uploaded yet</div>
                    <div style="font-size: 0.78rem; color: #475569; margin-top: 0.2rem;">Choose a JPG or PNG file to view preview</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Execution Action Bar
    exec_col1, exec_col2 = st.columns([2, 1])
    with exec_col1:
        analyze_btn = st.button("⚡ Run Azure AI Vision Analysis", type="primary", use_container_width=True)

    with exec_col2:
        if not is_azure_configured:
            st.warning("⚠️ Azure not configured. Set credentials in .env or try Offline Demo.")

    # Execute Analysis
    if analyze_btn:
        if not st.session_state.uploaded_image_bytes:
            st.warning("Please upload an image first before requesting analysis.")
        elif not is_azure_configured:
            st.error(
                "Cannot perform live analysis: Azure Vision credentials are not configured.\n\n"
                "To run live cloud analysis, configure AZURE_VISION_ENDPOINT and AZURE_VISION_KEY in `.env` "
                "(or in the Azure Configuration tab).\n\n"
                "Alternatively, switch to **'🏛️ Offline Demo Showcase'** in the sidebar to review the full interface with sample data."
            )
        else:
            with st.spinner("Communicating with Azure AI Vision 4.0 API (Captions, Objects, Tags, OCR)..."):
                result = vision_service.analyze_image(st.session_state.uploaded_image_bytes)

                if result.get("success"):
                    st.session_state.analysis_results = result
                    st.success("Azure AI Vision analysis completed successfully!")
                else:
                    st.error(f"Analysis Failed ({result.get('error_type')}): {result.get('message')}")
                    st.session_state.analysis_results = None

    # --------------------------------------------------------------------------
    # Render Analysis Results (if available)
    # --------------------------------------------------------------------------
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        render_disclaimer_banner(res.get("is_demo", False))

        # Hero Caption Banner
        render_caption_card(res["caption"]["text"], res["caption"]["confidence_pct"])

        # Key Metrics Grid
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(render_metric_card("Detected Objects", str(len(res["objects"])), "Visual entities found"), unsafe_allow_html=True)
        with m_col2:
            st.markdown(render_metric_card("Semantic Tags", str(len(res["tags"])), "Concept classification"), unsafe_allow_html=True)
        with m_col3:
            st.markdown(render_metric_card("OCR Text Lines", str(res["ocr"]["total_lines"]), f"{res['ocr']['total_words']} words read"), unsafe_allow_html=True)
        with m_col4:
            st.markdown(render_metric_card("Model Version", str(res.get("model_version", "2024-02-01")), "Azure Vision Engine"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs Section
        tab_overview, tab_objects, tab_tags, tab_captions, tab_ocr, tab_export = st.tabs([
            "📊 Overview",
            "🎯 Object Detection",
            "🏷️ Image Tags",
            "💬 Image Captions",
            "📝 OCR Results",
            "💾 Raw JSON & Downloads"
        ])

        # TAB 1: OVERVIEW
        with tab_overview:
            st.markdown("#### 🌟 Executive Analysis Summary")
            ov_col1, ov_col2 = st.columns([1, 1], gap="large")

            with ov_col1:
                st.markdown(
                    f"""
                    <div class="ai-card">
                        <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem; margin-bottom: 0.6rem;">
                            Primary Scene Interpretation
                        </div>
                        <p style="font-size: 0.9rem; color: #e2e8f0; line-height: 1.6;">
                            The Azure Vision 4.0 model classified this scene with high confidence.
                            Top descriptive tags and visual concepts detected include:
                        </p>
                        <div style="margin-top: 0.5rem;">
                            {''.join([f'<span class="tag-pill">{t["name"]}<span class="tag-pill-conf">{t["confidence_pct"]:.0f}%</span></span>' for t in res["tags"][:10]])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with ov_col2:
                st.markdown(
                    f"""
                    <div class="ai-card">
                        <div style="font-weight: 700; color: #10b981; font-size: 0.95rem; margin-bottom: 0.6rem;">
                            Object & Text Breakdown
                        </div>
                        <ul style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.7; padding-left: 1.2rem;">
                            <li><strong>Identified Objects:</strong> {len(res['objects'])} distinct objects located with bounding boxes.</li>
                            <li><strong>OCR Readable Text:</strong> {'Yes, readable text found' if res['ocr']['detected'] else 'No readable text detected'}.</li>
                            <li><strong>Dense Captions:</strong> {len(res.get('dense_captions', []))} contextual sub-region descriptions.</li>
                            <li><strong>People Detected:</strong> {len(res.get('people', []))} human subjects localized.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # TAB 2: OBJECT DETECTION
        with tab_objects:
            st.markdown("#### 🎯 Object Detection & Bounding Box Localization")

            filtered_objects = [o for o in res["objects"] if o["confidence"] >= min_conf_slider]

            obj_col_img, obj_col_table = st.columns([1.2, 1], gap="large")

            with obj_col_img:
                orig_img = Image.open(io.BytesIO(st.session_state.uploaded_image_bytes))
                annotated_img = annotate_objects_on_image(
                    image=orig_img,
                    detected_objects=filtered_objects,
                    min_confidence=min_conf_slider,
                    show_labels=show_box_labels,
                    show_confidence=show_box_conf,
                    line_thickness=box_thickness
                )
                st.image(
                    annotated_img,
                    caption=f"Visual Bounding Boxes (Showing {len(filtered_objects)} of {len(res['objects'])} objects at ≥ {min_conf_slider*100:.0f}% confidence)",
                    use_container_width=True
                )

                # Download annotated image button
                annotated_bytes = image_to_bytes(annotated_img)
                st.download_button(
                    label="📥 Download Annotated Image (PNG)",
                    data=annotated_bytes,
                    file_name="annotated_detection.png",
                    mime="image/png",
                    use_container_width=True
                )

            with obj_col_table:
                st.markdown(f"**Detected Entities Table ({len(filtered_objects)} items)**")
                if filtered_objects:
                    table_data = []
                    for idx, o in enumerate(filtered_objects):
                        bb = o["bounding_box"]
                        table_data.append({
                            "#": idx + 1,
                            "Object Name": o["name"].title(),
                            "Confidence": f"{o['confidence_pct']:.1f}%",
                            "Coordinates [X, Y, W, H]": f"[{bb['x']}, {bb['y']}, {bb['width']}, {bb['height']}]"
                        })
                    df_objects = pd.DataFrame(table_data)
                    st.dataframe(df_objects, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No objects detected above {min_conf_slider*100:.0f}% confidence threshold.")

        # TAB 3: IMAGE TAGS
        with tab_tags:
            st.markdown("#### 🏷️ Semantic Image Tags & Concept Taxonomy")
            st.caption("Tags identify objects, actions, setting types, colors, and contextual entities in the image.")

            tag_col1, tag_col2 = st.columns([1.2, 1], gap="large")

            with tag_col1:
                st.markdown("**Confidence Scores for Detected Tags**")
                for t in res["tags"]:
                    st.markdown(
                        render_confidence_bar_html(t["name"].title(), t["confidence_pct"]),
                        unsafe_allow_html=True
                    )

            with tag_col2:
                st.markdown("**Tag Cloud & Taxonomy**")
                tags_html = "".join([
                    f'<span class="tag-pill" style="font-size: 0.9rem; padding: 0.4rem 0.8rem;">{t["name"]} '
                    f'<span class="tag-pill-conf">{t["confidence_pct"]:.1f}%</span></span>'
                    for t in res["tags"]
                ])
                st.markdown(f'<div class="ai-card">{tags_html}</div>', unsafe_allow_html=True)

        # TAB 4: IMAGE CAPTIONING
        with tab_captions:
            st.markdown("#### 💬 Image Captioning & Dense Contextual Descriptions")
            st.caption("Azure AI Vision generates natural language sentences synthesizing visual relationships.")

            # Main Caption
            render_caption_card(res["caption"]["text"], res["caption"]["confidence_pct"])

            # Dense Captions Table
            st.markdown("##### 🔍 Dense Sub-Region Captions")
            if res.get("dense_captions"):
                dense_items = []
                for idx, dc in enumerate(res["dense_captions"]):
                    bb = dc["bounding_box"]
                    dense_items.append({
                        "#": idx + 1,
                        "Sub-Region Description": dc["text"],
                        "Confidence": f"{dc['confidence_pct']:.1f}%",
                        "Region Bounding Box": f"X:{bb['x']} Y:{bb['y']} W:{bb['width']} H:{bb['height']}"
                    })
                st.dataframe(pd.DataFrame(dense_items), use_container_width=True, hide_index=True)
            else:
                st.info("Dense captions not returned for this image.")

        # TAB 5: OCR RESULTS
        with tab_ocr:
            st.markdown("#### 📝 Optical Character Recognition (Read API)")
            st.caption("High-accuracy extraction of printed and handwritten text embedded inside the image.")

            if res["ocr"]["detected"]:
                ocr_col1, ocr_col2 = st.columns([1, 1], gap="large")

                with ocr_col1:
                    st.markdown(f"**Extracted Text ({res['ocr']['total_lines']} lines • {res['ocr']['total_words']} words)**")
                    # Copyable text area
                    st.text_area(
                        label="Copyable Extracted Text",
                        value=res["ocr"]["full_text"],
                        height=260,
                        label_visibility="collapsed"
                    )

                    # Download TXT button
                    st.download_button(
                        label="📥 Download Extracted Text (.txt)",
                        data=res["ocr"]["full_text"],
                        file_name="extracted_ocr_text.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                with ocr_col2:
                    st.markdown("**Structured Line-by-Line Breakdown**")
                    line_data = [
                        {"Line #": idx + 1, "Extracted Content": l["text"], "Words": l["words_count"]}
                        for idx, l in enumerate(res["ocr"]["lines"])
                    ]
                    st.dataframe(pd.DataFrame(line_data), use_container_width=True, hide_index=True)

            else:
                st.markdown(
                    """
                    <div class="ai-card" style="text-align: center; padding: 2.5rem;">
                        <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📭</div>
                        <div style="font-weight: 700; font-size: 1.1rem; color: #f1f5f9;">No Readable Text Detected</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">
                            Azure AI Vision's Read engine analyzed the image and found no printed or handwritten textual content.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # TAB 6: RAW JSON & DOWNLOADS
        with tab_export:
            st.markdown("#### 💾 Export & Machine-Readable Raw JSON")
            st.caption("Inspect and download full JSON responses matching Azure AI Vision 4.0 schema.")

            json_str = json.dumps(res["raw_json"], indent=2)

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.download_button(
                    label="📥 Download Full Analysis JSON",
                    data=json_str,
                    file_name="azure_vision_analysis.json",
                    mime="application/json",
                    use_container_width=True
                )
            with exp_col2:
                if res["ocr"]["detected"]:
                    st.download_button(
                        label="📥 Download Extracted OCR (.txt)",
                        data=res["ocr"]["full_text"],
                        file_name="extracted_ocr.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

            st.json(res["raw_json"])


# ------------------------------------------------------------------------------
# WORKSPACE 2: 🏛️ OFFLINE DEMO SHOWCASE
# ------------------------------------------------------------------------------
elif app_mode == "🏛️ Offline Demo Showcase":
    render_header(
        title="Offline Academic Demonstration Showcase",
        subtitle="Review full multi-modal UI capabilities using curated academic sample datasets (Zero Azure billing or cloud keys required).",
        is_live=False
    )

    st.markdown(
        """
        <div class="ai-card-accent" style="margin-bottom: 1.5rem;">
            <div style="font-weight: 700; color: #38bdf8; font-size: 1.05rem; margin-bottom: 0.3rem;">
                🎓 Offline Sample Demonstration
            </div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                Explore the interface without Azure credentials using curated sample images and pre-packaged example results.
                The captions, tags, OCR text, boxes, and scores are sample data—not analysis of your image and not a live Azure response.
                Image annotations and exports are rendered locally.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    demo_choice = st.radio(
        label="Select Academic Demonstration Scenario:",
        options=[
            "🚗 Preset 1: Urban Traffic & Object Detection (Street Scene)",
            "📊 Preset 2: Technical Conference Presentation & Architecture (OCR + Diagrams)",
            "🧾 Preset 3: Commercial Business Invoice (Document OCR + Data Extraction)"
        ],
        index=0,
        horizontal=True
    )

    preset_map = {
        "🚗 Preset 1: Urban Traffic & Object Detection (Street Scene)": ("street", "sample_street.jpg"),
        "📊 Preset 2: Technical Conference Presentation & Architecture (OCR + Diagrams)": ("presentation", "sample_presentation.png"),
        "🧾 Preset 3: Commercial Business Invoice (Document OCR + Data Extraction)": ("document", "sample_invoice.png")
    }

    preset_key, sample_file = preset_map[demo_choice]
    sample_img_path = os.path.join(ASSETS_DIR, sample_file)

    # Load demo analysis
    demo_result = vision_service.get_sample_demo_analysis(preset_key)

    if os.path.exists(sample_img_path):
        with open(sample_img_path, "rb") as f:
            demo_img_bytes = f.read()

        demo_col_img, demo_col_info = st.columns([1, 1], gap="large")

        with demo_col_img:
            st.image(Image.open(sample_img_path), caption=f"Demo Preset Image: {sample_file}", use_container_width=True)

        with demo_col_info:
            st.markdown(
                f"""
                <div class="ai-card">
                    <div style="font-weight: 700; color: #fbbf24; font-size: 0.95rem; margin-bottom: 0.5rem;">
                        ⚠️ Scenario Description: {demo_choice.split(':')[1].split('(')[0].strip()}
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                        <strong>Sample outputs shown:</strong><br>
                        • Example image caption and tags.<br>
                        • Example object labels and pixel coordinates.<br>
                        • Example OCR text and confidence values.<br>
                        No cloud service is contacted in this workspace.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Render Demo Disclaimer Banner
        render_disclaimer_banner(is_demo=True)

        # Hero Caption Banner
        render_caption_card(
            demo_result["caption"]["text"],
            demo_result["caption"]["confidence_pct"],
            is_demo=True
        )

        # Metric Cards
        dm1, dm2, dm3, dm4 = st.columns(4)
        with dm1:
            st.markdown(render_metric_card("Detected Objects", str(len(demo_result["objects"])), "Visual entities found"), unsafe_allow_html=True)
        with dm2:
            st.markdown(render_metric_card("Semantic Tags", str(len(demo_result["tags"])), "Concept taxonomy"), unsafe_allow_html=True)
        with dm3:
            st.markdown(render_metric_card("OCR Text Lines", str(demo_result["ocr"]["total_lines"]), f"{demo_result['ocr']['total_words']} words read"), unsafe_allow_html=True)
        with dm4:
            st.markdown(render_metric_card("Analysis Mode", "Academic Demo", "Pre-packaged sample"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Demo Tabs
        dtab_objects, dtab_tags, dtab_ocr, dtab_json = st.tabs([
            "🎯 Object Detection & Bounding Boxes",
            "🏷️ Image Tags & Taxonomy",
            "📝 OCR Text Extraction",
            "💾 Raw JSON & Downloads"
        ])

        with dtab_objects:
            st.markdown("#### 🎯 Object Detection with Pillow Bounding Boxes")
            filtered_demo_objs = [o for o in demo_result["objects"] if o["confidence"] >= min_conf_slider]

            d_col1, d_col2 = st.columns([1.2, 1], gap="large")
            with d_col1:
                orig_demo_img = Image.open(io.BytesIO(demo_img_bytes))
                ann_demo_img = annotate_objects_on_image(
                    image=orig_demo_img,
                    detected_objects=filtered_demo_objs,
                    min_confidence=min_conf_slider,
                    show_labels=show_box_labels,
                    show_confidence=show_box_conf,
                    line_thickness=box_thickness
                )
                st.image(ann_demo_img, caption=f"Annotated Demo Image ({len(filtered_demo_objs)} objects displayed)", use_container_width=True)

            with d_col2:
                st.markdown("**Detected Objects Table**")
                demo_obj_rows = [
                    {
                        "#": idx + 1,
                        "Object": o["name"].title(),
                        "Confidence": f"{o['confidence_pct']:.1f}%",
                        "Bounding Box [X, Y, W, H]": f"[{o['bounding_box']['x']}, {o['bounding_box']['y']}, {o['bounding_box']['width']}, {o['bounding_box']['height']}]"
                    }
                    for idx, o in enumerate(filtered_demo_objs)
                ]
                st.dataframe(pd.DataFrame(demo_obj_rows), use_container_width=True, hide_index=True)

        with dtab_tags:
            st.markdown("#### 🏷️ Semantic Tags & Visual Concept Breakdown")
            dt_col1, dt_col2 = st.columns([1.2, 1], gap="large")
            with dt_col1:
                for t in demo_result["tags"]:
                    st.markdown(render_confidence_bar_html(t["name"].title(), t["confidence_pct"]), unsafe_allow_html=True)
            with dt_col2:
                tags_html = "".join([
                    f'<span class="tag-pill" style="font-size: 0.9rem; padding: 0.4rem 0.8rem;">{t["name"]} '
                    f'<span class="tag-pill-conf">{t["confidence_pct"]:.1f}%</span></span>'
                    for t in demo_result["tags"]
                ])
                st.markdown(f'<div class="ai-card">{tags_html}</div>', unsafe_allow_html=True)

        with dtab_ocr:
            st.markdown("#### 📝 Optical Character Recognition (OCR)")
            if demo_result["ocr"]["detected"]:
                doc_col1, doc_col2 = st.columns([1, 1], gap="large")
                with doc_col1:
                    st.text_area("Extracted OCR Text Area", value=demo_result["ocr"]["full_text"], height=250)
                    st.download_button(
                        label="📥 Download Demo OCR Text (.txt)",
                        data=demo_result["ocr"]["full_text"],
                        file_name=f"{preset_key}_extracted_text.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with doc_col2:
                    d_lines = [
                        {"Line #": idx + 1, "Extracted Content": l["text"], "Words": l["words_count"]}
                        for idx, l in enumerate(demo_result["ocr"]["lines"])
                    ]
                    st.dataframe(pd.DataFrame(d_lines), use_container_width=True, hide_index=True)

        with dtab_json:
            st.markdown("#### 💾 Sample JSON Payload")
            demo_json = json.dumps(demo_result["raw_json"], indent=2)
            st.download_button(
                label="📥 Download Sample Analysis JSON",
                data=demo_json,
                file_name=f"{preset_key}_sample_analysis.json",
                mime="application/json",
                use_container_width=True
            )
            st.json(demo_result["raw_json"])


# ------------------------------------------------------------------------------
# WORKSPACE 3: 📖 ARCHITECTURE & AI-900 CONCEPTS
# ------------------------------------------------------------------------------
elif app_mode == "📖 Architecture & AI-900 Concepts":
    render_header(
        title="System Architecture & Azure AI Fundamentals (AI-900)",
        subtitle="Technical mapping between the application implementation and core Microsoft Azure AI curriculum concepts.",
        is_live=is_azure_configured
    )

    st.markdown(
        """
        <div class="ai-card" style="margin-bottom: 1.5rem;">
            <div style="font-weight: 700; color: #38bdf8; font-size: 1.1rem; margin-bottom: 0.4rem;">
                Academic Project Alignment: Microsoft Azure AI Fundamentals (AI-900)
            </div>
            <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.65;">
                This college project was designed and built to implement the core theoretical concepts tested in the
                <strong>Microsoft Certified: Azure AI Fundamentals (AI-900 / AI-901)</strong> examination.
                Rather than relying on local, compute-heavy deep learning libraries, the project demonstrates how
                enterprise software consumes pre-trained foundation models through Azure Cognitive Services / Azure AI Services.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    arch_tab1, arch_tab2, arch_tab3 = st.tabs([
        "🧠 Computer Vision Workloads",
        "🏗️ System & Cloud Architecture",
        "⚖️ Responsible AI Principles"
    ])

    with arch_tab1:
        st.markdown("### Core Computer Vision Workloads Implemented")
        st.markdown(
            """
            In the Azure AI curriculum, Computer Vision is defined as the capability of software systems to interpret
            and evaluate visual input from cameras, digital images, and video frames.

            | Workload | Azure AI Vision 4.0 Feature | Description | Implementation in this App |
            | :--- | :--- | :--- | :--- |
            | **Image Classification & Tagging** | `VisualFeatures.TAGS` | Assigns semantic category labels to an image based on recognized visual patterns. | Confidence-scored tag list and interactive taxonomy cloud. |
            | **Object Detection** | `VisualFeatures.OBJECTS` | Detects individual objects, identifies their class, and outputs coordinates $(x, y, w, h)$. | Pillow bounding box rendering with high-contrast color palettes. |
            | **Semantic Captioning** | `VisualFeatures.CAPTION` | Generates human-readable natural language sentences describing full scene context. | Prominent hero caption card powered by Microsoft Florence foundation model. |
            | **Dense Captioning** | `VisualFeatures.DENSE_CAPTIONS` | Generates localized captions for multiple distinctive regions within a single image. | Dense captions table with regional coordinates. |
            | **Optical Character Recognition** | `VisualFeatures.READ` | Detects and transcribes printed or handwritten text into machine-encoded strings. | Multi-line structured OCR viewer and copyable text area. |
            | **People Detection** | `VisualFeatures.PEOPLE` | Identifies human presence and bounding boxes for safety, retail, or access monitoring. | People count metric and localization overlay. |
            """
        )

    with arch_tab2:
        st.markdown("### End-to-End System Architecture")
        st.markdown(
            """
            ```
            ┌─────────────────────────────────────────────────────────────────────────────┐
            │                               CLIENT LAYER                                  │
            │  Streamlit Frontend (app.py) • Dark Themed UI • Responsive Grid             │
            │  - Image Upload & Validation (JPEG, PNG • Max 20MB • Min 50x50px)          │
            │  - Visualization Controller (Pillow Bounding Boxes, Confidence Thresholds)  │
            │  - Export Engine (JSON & TXT file downloads)                               │
            └──────────────────────────────┬──────────────────────────────────────────────┘
                                           │
                                           ▼
            ┌─────────────────────────────────────────────────────────────────────────────┐
            │                           LOCAL SERVICE LAYER                               │
            │  services/vision_service.py                                                 │
            │  - Credentials & Configuration Validator (.env)                             │
            │  - Connectivity Diagnostic Engine                                           │
            │  - Azure AI Vision 4.0 Client (SDK & Direct REST API Fallback)              │
            │  - Response Parser & Error Classifier                                       │
            └──────────────────────────────┬──────────────────────────────────────────────┘
                                           │ HTTPS (TLS 1.3 / Port 443)
                                           │ Ocp-Apim-Subscription-Key Header
                                           ▼
            ┌─────────────────────────────────────────────────────────────────────────────┐
            │                      MICROSOFT AZURE CLOUD LAYER                            │
            │  Azure AI Services / Computer Vision Resource                               │
            │  API Version: 2024-02-01                                                    │
            │  - Microsoft Florence Vision Foundation Model                               │
            │  - Multi-Modal Image Analysis 4.0 Engine                                    │
            │  - Read OCR Engine (Printed & Handwritten Text)                             │
            └─────────────────────────────────────────────────────────────────────────────┘
            ```
            """
        )

    with arch_tab3:
        st.markdown("### Responsible AI Principles (Microsoft Framework)")
        st.markdown(
            """
            Microsoft defines six foundational pillars for developing and deploying artificial intelligence systems.
            This project explicitly adheres to these principles:

            1. **Fairness:** The underlying Azure Vision models undergo rigorous debiasing across diverse demographic groups and global scenes.
            2. **Reliability & Safety:** Strict input validation ensures only valid, uncorrupted images are processed. Fallback mechanisms handle network faults gracefully.
            3. **Privacy & Security:** Sensitive API keys are stored exclusively in local `.env` configuration, excluded from version control via `.gitignore`, and masked in all UI representations. Raw images are not permanently cached or leaked.
            4. **Inclusiveness:** Dual input support allows analyzing both digital captures and physical paper documents through high-accuracy OCR.
            5. **Transparency:** Every AI interpretation provides explicit confidence percentages. The application never fakes live Azure calls and explicitly labels demo mode.
            6. **Accountability:** Software architecture separates UI, network services, and image processing, providing auditable error logging and testable modules.
            """
        )


# ------------------------------------------------------------------------------
# WORKSPACE 4: ⚙️ AZURE CONFIGURATION & DIAGNOSTICS
# ------------------------------------------------------------------------------
elif app_mode == "⚙️ Azure Configuration & Diagnostics":
    render_header(
        title="Azure Resource Configuration & Diagnostics",
        subtitle="Manage cloud credentials, test API latency, and review step-by-step Azure setup instructions.",
        is_live=is_azure_configured
    )

    conf_col1, conf_col2 = st.columns([1, 1], gap="large")

    with conf_col1:
        st.markdown("### 🔑 Active Credentials Status")

        endpoint_display = vision_service.endpoint or "None (Missing)"
        key_display = vision_service.get_masked_key()

        st.markdown(
            f"""
            <div class="ai-card">
                <div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 0.6rem;">
                    Configuration values currently loaded from <code>.env</code>:
                </div>
                <div style="margin-bottom: 0.8rem;">
                    <span style="font-size: 0.78rem; text-transform: uppercase; color: #64748b; font-weight: 700;">AZURE_VISION_ENDPOINT:</span><br>
                    <code style="font-size: 0.85rem; color: #38bdf8; word-break: break-all;">{endpoint_display}</code>
                </div>
                <div>
                    <span style="font-size: 0.78rem; text-transform: uppercase; color: #64748b; font-weight: 700;">AZURE_VISION_KEY:</span><br>
                    <code style="font-size: 0.85rem; color: #38bdf8;">{key_display}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 🧪 Live Diagnostics Test")
        st.markdown("Click below to send a lightweight diagnostic query to verify your endpoint and API key validity.")

        if st.button("🚀 Run Live Diagnostic Health Check", use_container_width=True):
            with st.spinner("Executing diagnostic handshake with Azure..."):
                diag = vision_service.test_connectivity()
                if diag["status"] == "success":
                    st.success(f"✅ {diag['message']}")
                    st.info(f"Target Endpoint: {diag['endpoint']}\nAPI Version: {diag.get('api_version')}")
                else:
                    st.error(f"❌ Diagnostic Failure ({diag['code']}):\n{diag['message']}")

    with conf_col2:
        st.markdown("### 📖 Step-by-Step Azure Resource Setup")
        st.markdown(
            """
            Follow these steps to set up your free Azure AI Vision resource:

            1. **Sign in to the Azure Portal:**
               Navigate to [portal.azure.com](https://portal.azure.com/).

            2. **Create a Resource:**
               - Click **Create a resource** in the top navigation.
               - Search for **Azure AI Services** (multi-service) or **Computer Vision** (single-service).
               - Select **Create**.

            3. **Configure Resource Details:**
               - **Subscription:** Select your active Azure subscription (or Azure for Students).
               - **Resource Group:** Select or create a resource group (e.g., `rg-vision-project`).
               - **Region:** Choose a region supporting Image Analysis 4.0 (e.g., `East US`, `West US 2`, `North Europe`).
               - **Name:** Choose a unique resource name (e.g., `vision-ai-demo-2026`).
               - **Pricing Tier:** Select **Free F0** (if available) or **Standard S0**.

            4. **Retrieve Keys and Endpoint:**
               - Once deployment completes, click **Go to resource**.
               - Under **Resource Management** in the left menu, click **Keys and Endpoint**.
               - Copy **KEY 1** and the **Endpoint** URL.

            5. **Update your local `.env` file:**
               ```env
               AZURE_VISION_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/
               AZURE_VISION_KEY=your_copied_key_1
               ```
            6. Restart or refresh the Streamlit application!
            """
        )

# Render unified footer
render_footer()
