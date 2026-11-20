"""
utils/ui_components.py
==============================================================================
UI/UX Theme and Component Library
------------------------------------------------------------------------------
Provides dark-themed styling, sleek cards, metric badges, and interactive
components for an enterprise-grade AI software product interface.
==============================================================================
"""

import streamlit as st


def inject_custom_theme():
    """
    Injects custom CSS to style Streamlit with a modern, high-contrast dark theme:
    near-black canvas (#0A0E17), charcoal cards (#141B2D), Azure blue accents (#0078D4, #38BDF8),
    refined typography, sleek tabs, and responsive layout.
    """
    custom_css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #162032;
        --bg-card-hover: #1c283f;
        --border-subtle: #223049;
        --border-focus: #0078D4;
        --azure-blue: #0078D4;
        --azure-light: #38BDF8;
        --azure-glow: rgba(0, 120, 212, 0.25);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-coral: #ef4444;
        --radius-lg: 12px;
        --radius-md: 8px;
    }

    /* Overall App Background & Typography */
    .stApp {
        background-color: var(--bg-main) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* Main Content Container Padding */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1350px !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d131f !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }

    /* Modern Card Container */
    .ai-card {
        background: linear-gradient(145deg, #131c2e, #182338);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .ai-card:hover {
        border-color: #2e4366;
    }

    .ai-card-accent {
        background: linear-gradient(145deg, #12213a, #162a4a);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: var(--radius-lg);
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 24px -2px rgba(0, 120, 212, 0.15);
    }

    /* Metric Cards Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.9rem;
        margin-bottom: 1.2rem;
    }
    .metric-box {
        background-color: #121a2a;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 1rem 1.1rem;
        text-align: left;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: var(--azure-light);
        margin-top: 0.2rem;
    }

    /* Badges & Tags */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-live {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-demo {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-error {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .tag-pill {
        display: inline-block;
        background-color: #182338;
        border: 1px solid #283a58;
        color: #e2e8f0;
        padding: 0.3rem 0.75rem;
        border-radius: 6px;
        font-size: 0.82rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        font-weight: 500;
    }
    .tag-pill-conf {
        color: var(--azure-light);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        margin-left: 0.3rem;
    }

    /* Caption Highlight Box */
    .caption-banner {
        background: linear-gradient(90deg, rgba(0, 120, 212, 0.18), rgba(56, 189, 248, 0.08));
        border-left: 4px solid var(--azure-blue);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 1.1rem 1.4rem;
        margin: 0.8rem 0 1.2rem 0;
    }
    .caption-text {
        font-size: 1.18rem;
        font-weight: 600;
        color: #ffffff;
        line-height: 1.5;
    }
    .caption-sub {
        font-size: 0.82rem;
        color: var(--azure-light);
        margin-top: 0.35rem;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #0e1522;
        padding: 0.35rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.6rem;
        padding: 0 1.2rem;
        background-color: transparent !important;
        border-radius: 6px !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0078D4, #0284c7) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(0, 120, 212, 0.35) !important;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0078D4 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1px solid #38bdf8 !important;
        border-radius: var(--radius-md) !important;
        padding: 0.55rem 1.3rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 120, 212, 0.25) !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0284c7 0%, #0078D4 100%) !important;
        box-shadow: 0 4px 16px rgba(0, 120, 212, 0.45) !important;
        transform: translateY(-1px);
    }

    /* Download Buttons */
    div.stDownloadButton > button {
        background: #142033 !important;
        color: #38bdf8 !important;
        border: 1px solid #283e60 !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
    }
    div.stDownloadButton > button:hover {
        background: #1c2c47 !important;
        border-color: #38bdf8 !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #0e1522;
        border: 2px dashed #243550;
        border-radius: var(--radius-lg);
        padding: 1.2rem;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--azure-blue);
    }

    /* OCR Code / Text Area */
    .ocr-display-box {
        background-color: #0d121c;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: #e2e8f0;
        line-height: 1.65;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }

    /* Confidence Progress Bars */
    .conf-bar-container {
        margin: 0.45rem 0;
    }
    .conf-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        margin-bottom: 0.2rem;
        color: var(--text-primary);
    }
    .conf-bar-bg {
        background-color: #1a2538;
        border-radius: 9999px;
        height: 8px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #0078D4, #38BDF8);
    }

    /* Table Styling */
    .dataframe {
        background-color: #121927 !important;
        color: #f1f5f9 !important;
        border: 1px solid var(--border-subtle) !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(
    title: str = "AI-Based Image Recognition & Analysis System",
    subtitle: str = "Enterprise Cloud Vision Multi-Modal Analysis Platform • Microsoft Azure AI Vision 4.0",
    is_live: bool = False
):
    """Renders the top application header with title, subtitle, and live badge."""
    badge_html = (
        '<span class="status-badge badge-live">● Azure Vision Live</span>'
        if is_live else
        '<span class="status-badge badge-demo">● Demo / Offline Mode</span>'
    )

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.4rem;">
            <div>
                <h1 style="font-size: 2.1rem; font-weight: 800; margin: 0; color: #ffffff; letter-spacing: -0.02em;">
                    {title}
                </h1>
                <p style="font-size: 0.95rem; color: #94a3b8; margin: 0.35rem 0 0 0;">
                    {subtitle}
                </p>
            </div>
            <div style="margin-top: 0.5rem;">
                {badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_disclaimer_banner(is_demo: bool):
    """Renders a mandatory clear notice if running in demo or live mode."""
    if is_demo:
        st.markdown(
            """
            <div style="background-color: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem;">
                <span style="font-size: 1.2rem;">⚠️</span>
                <div style="font-size: 0.85rem; color: #fbbf24;">
                    <strong>ACADEMIC DEMO MODE ACTIVE:</strong> Azure AI Vision credentials are not configured or preset was selected.
                    Showing pre-packaged sample analysis for UI and visualization testing. <em>Azure cloud was not billed or contacted.</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_caption_card(caption_text: str, confidence_pct: float):
    """Renders a prominent image caption hero banner."""
    st.markdown(
        f"""
        <div class="caption-banner">
            <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #38bdf8; font-weight: 700; margin-bottom: 0.3rem;">
                Natural Language Caption (Azure Vision 4.0)
            </div>
            <div class="caption-text">
                "{caption_text}"
            </div>
            <div class="caption-sub">
                Model Confidence: <strong>{confidence_pct:.2f}%</strong> • Generated via Microsoft Florence-backed Vision Foundation Model
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(label: str, value: str, subtext: str = "") -> str:
    """Returns HTML for a single modern metric box."""
    return f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub">{subtext}</div>' if subtext else ''}
    </div>
    """


def render_confidence_bar_html(label: str, confidence_pct: float, color_gradient: str = None) -> str:
    """Returns HTML for an individual confidence progress bar."""
    gradient = color_gradient or "linear-gradient(90deg, #0078D4, #38BDF8)"
    return f"""
    <div class="conf-bar-container">
        <div class="conf-bar-label">
            <span style="font-weight: 500;">{label}</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #38bdf8;">{confidence_pct:.1f}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width: {min(max(confidence_pct, 0.0), 100.0)}%; background: {gradient};"></div>
        </div>
    </div>
    """


def render_footer():
    """Renders academic project attribution footer."""
    st.markdown(
        """
        <div style="margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid #1a2538; text-align: center; color: #64748b; font-size: 0.82rem;">
            <p style="margin: 0;">
                <strong>AI-Based Image Recognition and Analysis System</strong> • BE Computer Science Engineering (AI & ML)
            </p>
            <p style="margin: 0.3rem 0 0 0;">
                Aligned with Microsoft Azure AI Fundamentals (AI-900 / AI-901) • Powered by Azure AI Vision 4.0 Image Analysis & Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
