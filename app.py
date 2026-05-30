"""
Production Streamlit interface for ClinIQ.

ClinIQ is an NHS-themed Retrieval-Augmented Generation chat app for querying
NICE clinical guideline PDFs through the existing backend pipeline.
"""

import html
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components

from src.ingestion import load_existing_vector_store
from src.pipeline import ClinIQPipeline

load_dotenv()

# Inline CSS for st.components.v1.html iframes (CSS variables not available inside iframes)
_STATS_IFRAME_CSS = """
body { margin: 0; padding: 0; background: transparent; }
.clq-stats-bar {
    display: flex; align-items: center; gap: 0.65rem; flex-wrap: wrap;
    border-top: 1px solid #2A3F5F; border-bottom: 1px solid #2A3F5F;
    padding: 0.55rem 0; color: #8B9BB4;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
}
.clq-stat-divider { display: inline-block; width: 1px; height: 14px; background: #2A3F5F; }
"""

_SOURCES_IFRAME_CSS = """
body { margin: 0; padding: 0; background: transparent; font-family: 'Inter', sans-serif; color: #F0F4F8; }
details { border-radius: 8px; }
summary { cursor: pointer; color: #F0F4F8; font-size: 13px; font-weight: 500; margin-bottom: 0.65rem; }
.clq-source-card {
    position: relative; background: #243550; border: 1px solid #2A3F5F;
    border-radius: 8px; padding: 0.8rem 0.85rem 0.8rem 1rem;
    margin-bottom: 0.65rem; overflow: hidden;
}
.clq-source-card::before {
    content: ''; position: absolute; inset: 0 auto 0 0; width: 4px; background: #005EB8;
}
.clq-source-title { color: #F0F4F8; font-size: 13px; font-weight: 600; margin-bottom: 0.25rem; }
.clq-source-meta { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.35rem; }
.clq-source-ref { color: #00D4FF; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.clq-source-page { color: #8B9BB4; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.clq-source-excerpt { color: #8B9BB4; font-size: 12px; font-style: italic; line-height: 1.5; }
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ClinIQ - NHS Guidelines Intelligence",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    """Inject the complete ClinIQ design system and Streamlit overrides."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --cliniq-navy: #0A1628;
            --cliniq-surface: #1E2D45;
            --cliniq-surface2: #243550;
            --cliniq-blue: #005EB8;
            --cliniq-cyan: #00D4FF;
            --cliniq-text: #F0F4F8;
            --cliniq-muted: #8B9BB4;
            --cliniq-success: #00C48C;
            --cliniq-warning: #FFB020;
            --cliniq-danger: #FF4757;
            --cliniq-border: #2A3F5F;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.3); }
            50% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.8); }
        }

        @keyframes typingDot {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }

        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(30px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        @keyframes confidenceRingFill {
            from { stroke-dashoffset: 100; }
        }

        #MainMenu, header, footer, [data-testid="stToolbar"],
        [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
            display: none !important;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--cliniq-navy) !important;
            color: var(--cliniq-text) !important;
            font-family: 'Inter', sans-serif !important;
        }

        [data-testid="stSidebar"] {
            width: 320px !important;
            min-width: 320px !important;
            max-width: 320px !important;
            background: var(--cliniq-surface) !important;
            border-right: 1px solid var(--cliniq-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 2rem 1.2rem !important;
        }

        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 900px;
            padding: 2.5rem 1.25rem 7rem;
        }

        h1, h2, h3, p, span, div, button, textarea {
            font-family: 'Inter', sans-serif;
            letter-spacing: 0;
        }

        .clq-sidebar-logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.15rem;
        }

        .clq-logo-cross {
            width: 34px;
            height: 34px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            color: var(--cliniq-cyan);
            border: 1px solid rgba(0, 212, 255, 0.55);
            background: rgba(0, 212, 255, 0.08);
            animation: pulseGlow 2s infinite;
            font-size: 21px;
            font-weight: 700;
        }

        .clq-logo-title {
            color: var(--cliniq-text);
            font-size: 28px;
            font-weight: 700;
            line-height: 1;
        }

        .clq-subtitle {
            color: var(--cliniq-muted);
            font-size: 13px;
            font-weight: 300;
            margin-bottom: 1.4rem;
        }

        .clq-divider {
            height: 1px;
            background: var(--cliniq-border);
            margin: 1.2rem 0;
        }

        .clq-status-card,
        .clq-step-card,
        .clq-disclaimer,
        .clq-welcome-card,
        .clq-assistant-bubble {
            background: var(--cliniq-surface);
            border: 1px solid var(--cliniq-border);
        }

        .clq-status-card {
            border-radius: 8px;
            padding: 0.9rem;
            animation: fadeInUp 0.4s ease-out;
        }

        .clq-ready-row {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--cliniq-text);
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 0.55rem;
        }

        .clq-ready-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--cliniq-success);
            animation: pulseGlow 2s infinite;
        }

        .clq-mono {
            color: var(--cliniq-muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            line-height: 1.75;
        }

        .clq-section-title {
            color: var(--cliniq-muted);
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            margin: 0 0 0.75rem;
        }

        .clq-step-card {
            display: grid;
            grid-template-columns: 24px 1fr;
            gap: 0.75rem;
            border-radius: 8px;
            padding: 0.8rem;
            margin-bottom: 0.7rem;
            animation: fadeInUp 0.4s ease-out both;
        }

        .clq-step-card:nth-child(2) { animation-delay: 0.1s; }
        .clq-step-card:nth-child(3) { animation-delay: 0.2s; }
        .clq-step-card:nth-child(4) { animation-delay: 0.3s; }

        .clq-step-badge {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: var(--cliniq-blue);
            color: white;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 700;
        }

        .clq-step-title {
            color: var(--cliniq-text);
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 0.15rem;
        }

        .clq-step-copy {
            color: var(--cliniq-muted);
            font-size: 12px;
            font-weight: 400;
            line-height: 1.45;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            min-height: 40px;
            background: var(--cliniq-surface2);
            border: 1px solid var(--cliniq-border);
            border-radius: 999px;
            color: var(--cliniq-text);
            font-size: 13px;
            font-weight: 400;
            text-align: left;
            justify-content: flex-start;
            transition: all 0.2s ease;
            margin-bottom: 0.35rem;
        }

        [data-testid="stSidebar"] .stButton > button::before {
            content: '→';
            color: var(--cliniq-cyan);
            margin-right: 0.45rem;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: var(--cliniq-cyan);
            background: #2b3f60;
            animation: pulseGlow 2s infinite;
            color: var(--cliniq-text);
        }

        [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:has(.clq-clear-marker) + div .stButton > button,
        [data-testid="stSidebar"] .clq-clear-button button {
            background: transparent !important;
            border-color: var(--cliniq-danger) !important;
            color: var(--cliniq-danger) !important;
            justify-content: center;
        }

        [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:has(.clq-clear-marker) + div .stButton > button::before {
            content: '' !important;
            margin: 0 !important;
        }

        [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:has(.clq-clear-marker) + div .stButton > button:hover {
            background: rgba(255, 71, 87, 0.1) !important;
            animation: none;
        }

        .clq-disclaimer {
            border-left: 4px solid var(--cliniq-warning);
            border-radius: 8px;
            padding: 0.85rem;
            color: var(--cliniq-muted);
            font-size: 11px;
            font-weight: 300;
            line-height: 1.55;
        }

        .clq-hero {
            animation: fadeInUp 0.4s ease-out;
            margin-bottom: 2.5rem;
        }

        .clq-hero h1 {
            color: var(--cliniq-text);
            font-size: 42px;
            font-weight: 700;
            margin: 0;
        }

        .clq-hero p {
            color: var(--cliniq-muted);
            font-size: 16px;
            font-weight: 300;
            margin: 0.35rem 0 1rem;
        }

        .clq-gradient-divider {
            height: 1px;
            width: 100%;
            background: linear-gradient(90deg, var(--cliniq-blue), transparent);
        }

        .clq-welcome-card {
            max-width: 620px;
            margin: 4rem auto 0;
            border-radius: 8px;
            padding: 2.2rem;
            text-align: center;
            animation: fadeInUp 0.4s ease-out;
        }

        .clq-welcome-icon {
            color: var(--cliniq-cyan);
            font-size: 48px;
            line-height: 1;
            margin-bottom: 0.8rem;
        }

        .clq-welcome-title {
            color: var(--cliniq-text);
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 1.3rem;
        }

        .clq-feature-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.65rem;
        }

        .clq-feature-pill {
            background: var(--cliniq-surface2);
            border: 1px solid var(--cliniq-border);
            border-radius: 999px;
            color: var(--cliniq-muted);
            font-size: 12px;
            font-weight: 400;
            padding: 0.45rem 0.7rem;
        }

        .clq-chat-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            padding-top: 0.75rem;
        }

        .clq-message-row {
            display: flex;
            width: 100%;
            animation: fadeInUp 0.4s ease-out;
        }

        .clq-user-row { justify-content: flex-end; }
        .clq-assistant-row { justify-content: flex-start; }

        .clq-message-stack {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .clq-user-stack {
            align-items: flex-end;
            max-width: 75%;
        }

        .clq-assistant-stack {
            align-items: flex-start;
            max-width: 85%;
        }

        .clq-user-bubble {
            background: var(--cliniq-blue);
            color: white;
            border-radius: 18px 18px 4px 18px;
            padding: 12px 16px;
            font-size: 15px;
            font-weight: 400;
            line-height: 1.55;
        }

        .clq-assistant-bubble {
            border-radius: 4px 18px 18px 18px;
            padding: 16px 20px;
            color: var(--cliniq-text);
            font-size: 15px;
            font-weight: 400;
            line-height: 1.7;
        }

        .clq-timestamp {
            color: var(--cliniq-muted);
            font-size: 11px;
            font-weight: 400;
            padding: 0 0.25rem;
        }

        .clq-answer-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.7rem;
        }

        .clq-answer-label {
            color: var(--cliniq-cyan);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
        }

        .clq-answer-text {
            color: var(--cliniq-text);
            font-size: 15px;
            line-height: 1.7;
        }

        .clq-answer-text p {
            margin: 0 0 0.75rem;
        }

        .clq-stats-bar {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            flex-wrap: wrap;
            border-top: 1px solid var(--cliniq-border);
            border-bottom: 1px solid var(--cliniq-border);
            margin: 1rem 0;
            padding: 0.55rem 0;
            color: var(--cliniq-muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }

        .clq-stat-divider {
            width: 1px;
            height: 14px;
            background: var(--cliniq-border);
        }

        .clq-sources {
            margin-top: 0.5rem;
        }

        .clq-sources details {
            border-radius: 8px;
        }

        .clq-sources summary {
            cursor: pointer;
            color: var(--cliniq-text);
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 0.65rem;
        }

        .clq-source-card {
            position: relative;
            background: var(--cliniq-surface2);
            border: 1px solid var(--cliniq-border);
            border-radius: 8px;
            padding: 0.8rem 0.85rem 0.8rem 1rem;
            margin-bottom: 0.65rem;
            overflow: hidden;
            animation: slideInRight 0.3s ease-out both;
        }

        .clq-source-card::before {
            content: '';
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: var(--cliniq-blue);
        }

        .clq-source-title {
            color: var(--cliniq-text);
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .clq-source-meta {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-bottom: 0.35rem;
        }

        .clq-source-ref {
            color: var(--cliniq-cyan);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }

        .clq-source-page {
            color: var(--cliniq-muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }

        .clq-source-excerpt {
            color: var(--cliniq-muted);
            font-size: 12px;
            font-weight: 300;
            font-style: italic;
            line-height: 1.5;
        }

        .clq-retrieval-bar {
            border-radius: 999px;
            padding: 0.55rem 0.8rem;
            margin-bottom: 0.8rem;
            color: var(--cliniq-text);
            font-size: 13px;
            background: linear-gradient(
                90deg,
                rgba(0, 94, 184, 0.25) 0%,
                rgba(0, 212, 255, 0.2) 50%,
                rgba(0, 94, 184, 0.25) 100%
            );
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        }

        .clq-typing-wrap {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .clq-typing-dots {
            display: flex;
            gap: 0.35rem;
            align-items: center;
        }

        .clq-typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--cliniq-cyan);
            animation: typingDot 1.2s infinite;
        }

        .clq-typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .clq-typing-dot:nth-child(3) { animation-delay: 0.4s; }

        .clq-typing-copy {
            color: var(--cliniq-muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }

        .stChatInput {
            background: transparent !important;
        }

        [data-testid="stChatInput"] {
            background: var(--cliniq-surface2) !important;
            border: 1px solid var(--cliniq-border) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: var(--cliniq-cyan) !important;
            box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.15) !important;
        }

        [data-testid="stChatInput"] textarea {
            color: var(--cliniq-text) !important;
            font-size: 15px !important;
            font-weight: 400 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--cliniq-muted) !important;
        }

        [data-testid="stChatInput"] button {
            background: var(--cliniq-blue) !important;
            color: white !important;
            border-radius: 8px !important;
        }

        @media (max-width: 760px) {
            [data-testid="stSidebar"] {
                width: 100% !important;
                min-width: 100% !important;
                max-width: 100% !important;
            }

            [data-testid="stAppViewContainer"] .main .block-container {
                padding: 1.2rem 0.8rem 7rem;
            }

            .clq-user-stack,
            .clq-assistant-stack {
                max-width: 94%;
            }

            .clq-hero h1 {
                font-size: 34px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """Initialize Streamlit state for chat, pipeline, and UI metadata."""
    defaults = {
        "messages": [],
        "pipeline": None,
        "vector_store_loaded": False,
        "indexed_chunks": 1936,
        "guideline_count": 5,
        "pending_question": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_vector_store_and_pipeline():
    """Load the existing Chroma vector store and create the unchanged backend pipeline."""
    if st.session_state.vector_store_loaded and st.session_state.pipeline is not None:
        return True

    try:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        docs_dir = Path(os.getenv("NHS_DOCS_DIR", "./data/nhs_docs"))

        if not Path(persist_dir).exists():
            st.error(
                "Vector store not found. Run `python ingest.py` after placing NICE PDFs "
                "in `data/nhs_docs/`, then refresh ClinIQ."
            )
            return False

        logger.info("Loading vector store from %s", persist_dir)
        vector_store = load_existing_vector_store(persist_dir)

        try:
            st.session_state.indexed_chunks = vector_store._collection.count()
        except Exception:
            logger.info("Could not read Chroma collection count; using configured fallback")

        if docs_dir.exists():
            st.session_state.guideline_count = len(list(docs_dir.glob("*.pdf"))) or 5

        logger.info("Initializing ClinIQ pipeline")
        st.session_state.pipeline = ClinIQPipeline(vector_store, use_weave=True)
        st.session_state.vector_store_loaded = True
        return True

    except Exception as exc:
        logger.error("Error loading vector store and pipeline: %s", exc, exc_info=True)
        st.error(f"Could not initialize ClinIQ: {exc}")
        return False


def render_typing_indicator(placeholder):
    """Render animated retrieval and typing state into a placeholder."""
    chunks = f"{st.session_state.indexed_chunks:,}"
    with placeholder.container():
        st.markdown(
            f"""
            <div class="clq-retrieval-bar">Searching {chunks} chunks...</div>
            <div class="clq-typing-wrap">
              <div class="clq-typing-dots">
                <span class="clq-typing-dot"></span>
                <span class="clq-typing-dot"></span>
                <span class="clq-typing-dot"></span>
              </div>
              <span class="clq-typing-copy">retrieving guideline evidence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_confidence_ring(score):
    """Return an inline SVG confidence ring for the proxy retrieval score.

    Uses hardcoded hex values so the SVG renders correctly inside an iframe
    (st.components.v1.html) where CSS variables from the parent page are absent.
    """
    safe_score = max(0, min(100, int(score)))
    dash_offset = 100 - safe_score
    if safe_score > 70:
        color = "#00C48C"
    elif safe_score >= 40:
        color = "#FFB020"
    else:
        color = "#FF4757"

    return f"""
    <svg width="40" height="40" viewBox="0 0 40 40" role="img" aria-label="Confidence {safe_score}%">
      <circle cx="20" cy="20" r="15.9" fill="none" stroke="#2A3F5F" stroke-width="4"></circle>
      <circle cx="20" cy="20" r="15.9" fill="none" stroke="{color}" stroke-width="4"
        stroke-dasharray="100" stroke-dashoffset="{dash_offset}"
        stroke-linecap="round" transform="rotate(-90 20 20)"
        style="animation: confidenceRingFill 1s ease-out;"></circle>
      <text x="20" y="23" text-anchor="middle" fill="#F0F4F8"
        font-family="JetBrains Mono, monospace" font-size="9" font-weight="500">{safe_score}%</text>
    </svg>
    """


def render_source_cards(sources, chunks):
    """Return collapsible source cards with NICE refs, pages, and retrieved excerpts."""
    if not sources:
        return """
        <div class="clq-sources">
          <details>
            <summary>Sources (0)</summary>
            <div class="clq-source-excerpt">No guideline citations were returned for this response.</div>
          </details>
        </div>
        """

    chunk_lookup = {}
    for chunk in chunks or []:
        if not hasattr(chunk, "metadata"):
            continue
        source = Path(str(chunk.metadata.get("source", "Unknown source"))).name
        page = str(chunk.metadata.get("page", "Unknown page"))
        excerpt = " ".join(chunk.page_content.split())[:150]
        chunk_lookup.setdefault((source, page), excerpt)

    cards = []
    for index, source_info in enumerate(sources):
        source = Path(str(source_info.get("source", "Unknown source"))).name
        page = str(source_info.get("page", "Unknown page"))
        nice_ref = _extract_nice_reference(source)
        title = _format_guideline_name(source)
        excerpt = source_info.get("excerpt") or chunk_lookup.get((source, page), "")
        if not excerpt:
            excerpt = "Retrieved from the indexed NICE guideline context for this answer."

        cards.append(
            f"""
            <div class="clq-source-card" style="animation-delay: {index * 0.1}s;">
              <div class="clq-source-title">{html.escape(title)}</div>
              <div class="clq-source-meta">
                <span class="clq-source-ref">{html.escape(nice_ref)}</span>
                <span class="clq-source-page">Page {html.escape(page)}</span>
              </div>
              <div class="clq-source-excerpt">{html.escape(excerpt)}</div>
            </div>
            """
        )

    return f"""
    <div class="clq-sources">
      <details>
        <summary>Sources ({len(sources)})</summary>
        {''.join(cards)}
      </details>
    </div>
    """


def render_stats_bar(response_time, chunks_retrieved, total_tokens):
    """Return the query stats bar."""
    return f"""
    <div class="clq-stats-bar">
      <span>{int(response_time)}ms</span>
      <span class="clq-stat-divider"></span>
      <span>{int(chunks_retrieved)} chunks</span>
      <span class="clq-stat-divider"></span>
      <span>{int(total_tokens)} tokens</span>
    </div>
    """


def _display_user_message(content, timestamp):
    """Render a user message using the native chat_message context manager."""
    with st.chat_message("user"):
        st.markdown(content)
        st.caption(timestamp)


def _display_assistant_message(
    content,
    confidence,
    response_time,
    chunks_retrieved,
    total_tokens,
    sources,
    timestamp,
    chunks=None,
):
    """Render an assistant response using layered Streamlit primitives.

    - st.markdown() with unsafe_allow_html for minimal structural wrappers only.
    - st.markdown() without unsafe_allow_html for answer text (natural markdown).
    - st.components.v1.html() for SVG ring, stats bar, and source cards.
    """
    with st.chat_message("assistant"):
        st.markdown('<span class="clq-answer-label">Guideline answer</span>', unsafe_allow_html=True)

        ring_svg = render_confidence_ring(confidence)
        components.html(
            f"<style>body{{margin:0;background:transparent;}} "
            f"@keyframes confidenceRingFill{{from{{stroke-dashoffset:100;}}}}</style>"
            f"{ring_svg}",
            height=50,
        )

        with st.container():
            st.markdown(content)

        components.html(
            f"<style>{_STATS_IFRAME_CSS}</style>"
            f"{render_stats_bar(response_time, chunks_retrieved, total_tokens)}",
            height=40,
        )

        source_cards_html = render_source_cards(sources, chunks or [])
        card_height = max(80, len(sources) * 120 + 60) if sources else 80
        components.html(
            f"<style>{_SOURCES_IFRAME_CSS}</style>{source_cards_html}",
            height=card_height,
        )

        st.caption(timestamp)


def render_sidebar():
    """Render the custom ClinIQ sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div class="clq-sidebar-logo">
              <span class="clq-logo-cross">✚</span>
              <span class="clq-logo-title">ClinIQ</span>
            </div>
            <div class="clq-subtitle">NHS Guidelines Intelligence</div>
            <div class="clq-divider"></div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.vector_store_loaded:
            st.markdown(
                f"""
                <div class="clq-status-card">
                  <div class="clq-ready-row">
                    <span class="clq-ready-dot"></span>
                    <span>System Ready</span>
                  </div>
                  <div class="clq-mono">{st.session_state.indexed_chunks:,} chunks indexed</div>
                  <div class="clq-mono">{st.session_state.guideline_count} NHS guidelines loaded</div>
                </div>
                <div class="clq-divider"></div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="clq-section-title">How it works</div>
            <div class="clq-step-card">
              <span class="clq-step-badge">1</span>
              <div>
                <div class="clq-step-title">Ask</div>
                <div class="clq-step-copy">Enter a clinical guideline question in plain language.</div>
              </div>
            </div>
            <div class="clq-step-card">
              <span class="clq-step-badge">2</span>
              <div>
                <div class="clq-step-title">Retrieve</div>
                <div class="clq-step-copy">ClinIQ searches indexed NICE guideline chunks.</div>
              </div>
            </div>
            <div class="clq-step-card">
              <span class="clq-step-badge">3</span>
              <div>
                <div class="clq-step-title">Answer</div>
                <div class="clq-step-copy">GPT-4o-mini writes a sourced response.</div>
              </div>
            </div>
            <div class="clq-divider"></div>
            <div class="clq-section-title">Try asking...</div>
            """,
            unsafe_allow_html=True,
        )

        example_questions = [
            "What are the NICE guidelines for hypertension?",
            "How should type 2 diabetes be managed?",
            "What is the recommended treatment for depression?",
            "How should asthma be monitored?",
        ]
        for index, question in enumerate(example_questions):
            if st.button(question, key=f"example_question_{index}"):
                st.session_state.pending_question = question

        st.markdown(
            """
            <div class="clq-divider"></div>
            <span class="clq-clear-marker"></span>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Clear conversation", key="clear_conversation"):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

        st.markdown(
            """
            <div class="clq-divider"></div>
            <div class="clq-disclaimer">
              <strong>⚕️ Clinical safety</strong><br>
              ClinIQ summarizes indexed NHS/NICE guideline text only. It is not a
              substitute for professional medical advice, diagnosis, or treatment.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_welcome_state():
    """Render the no-message hero and welcome card."""
    st.markdown(
        f"""
        <section class="clq-hero">
          <h1>ClinIQ</h1>
          <p>Intelligent NHS Clinical Guidelines Assistant</p>
          <div class="clq-gradient-divider"></div>
        </section>
        <section class="clq-welcome-card">
          <div class="clq-welcome-icon">✚</div>
          <div class="clq-welcome-title">Ask me anything about NHS clinical guidelines</div>
          <div class="clq-feature-row">
            <span class="clq-feature-pill">📋 {st.session_state.guideline_count} Guidelines</span>
            <span class="clq-feature-pill">🔍 {st.session_state.indexed_chunks:,} Chunks</span>
            <span class="clq-feature-pill">⚡ GPT-4o-mini</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def display_messages():
    """Render all messages from session state using Streamlit-native primitives."""
    if not st.session_state.messages:
        render_welcome_state()
        return

    for message in st.session_state.messages:
        if message["role"] == "user":
            _display_user_message(message["content"], message["timestamp"])
        else:
            _display_assistant_message(
                message["content"],
                message.get("confidence", 20),
                message.get("response_time", 0),
                message.get("chunks_retrieved", 0),
                message.get("total_tokens", 0),
                message.get("sources", []),
                message.get("timestamp", ""),
            )


def process_user_query(user_input):
    """Run a query, show typing state, and append the full message structure."""
    question = user_input.strip()
    if not question:
        return

    user_timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "timestamp": user_timestamp,
        }
    )
    _display_user_message(question, user_timestamp)

    placeholder = st.empty()
    render_typing_indicator(placeholder)

    retrieved_chunks = []
    start_time = time.time()
    try:
        retrieved_chunks = st.session_state.pipeline.retriever.invoke(question)
        response = st.session_state.pipeline.query(question)
        response_time = int((time.time() - start_time) * 1000)

        sources = _enrich_sources(response.get("sources", []), retrieved_chunks)
        confidence = _calculate_confidence(sources)
        answer = response.get("answer", "No answer was generated.")
        total_tokens = _estimate_total_tokens(question, answer, retrieved_chunks)
        timestamp = datetime.now().strftime("%H:%M")

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confidence": confidence,
            "response_time": response_time,
            "chunks_retrieved": len(retrieved_chunks),
            "total_tokens": total_tokens,
            "timestamp": timestamp,
        }
        st.session_state.messages.append(assistant_message)
        placeholder.empty()
        _display_assistant_message(
            answer,
            confidence,
            response_time,
            len(retrieved_chunks),
            total_tokens,
            sources,
            timestamp,
            chunks=retrieved_chunks,
        )

    except Exception as exc:
        logger.error("Error generating response: %s", exc, exc_info=True)
        response_time = int((time.time() - start_time) * 1000)
        timestamp = datetime.now().strftime("%H:%M")
        error_message = f"ClinIQ could not generate a response: {exc}"
        assistant_message = {
            "role": "assistant",
            "content": error_message,
            "sources": [],
            "confidence": 0,
            "response_time": response_time,
            "chunks_retrieved": len(retrieved_chunks),
            "total_tokens": 0,
            "timestamp": timestamp,
        }
        st.session_state.messages.append(assistant_message)
        placeholder.empty()
        _display_assistant_message(
            error_message,
            0,
            response_time,
            len(retrieved_chunks),
            0,
            [],
            timestamp,
        )


def main():
    """Main Streamlit entry point."""
    inject_css()
    initialize_session_state()

    load_vector_store_and_pipeline()
    render_sidebar()

    if not st.session_state.vector_store_loaded:
        st.stop()

    display_messages()

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        process_user_query(question)

    user_input = st.chat_input("Ask me about NHS clinical guidelines...")
    if user_input:
        process_user_query(user_input)


def _extract_nice_reference(filename):
    """Extract a NICE reference such as NG136 from a source filename."""
    match = re.search(r"(NG\d+)", filename.upper())
    return match.group(1) if match else "NICE"


def _format_guideline_name(filename):
    """Turn a PDF filename into a readable guideline title."""
    stem = Path(filename).stem
    cleaned = re.sub(r"^ng\d+[_-]?", "", stem, flags=re.IGNORECASE)
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned else filename


def _enrich_sources(sources, chunks):
    """Add filename normalization, NICE refs, and excerpts to source dictionaries."""
    chunk_lookup = {}
    for chunk in chunks:
        source = Path(str(chunk.metadata.get("source", "Unknown source"))).name
        page = str(chunk.metadata.get("page", "Unknown page"))
        excerpt = " ".join(chunk.page_content.split())[:150]
        chunk_lookup.setdefault((source, page), excerpt)

    enriched = []
    seen = set()
    for source_info in sources:
        source = Path(str(source_info.get("source", "Unknown source"))).name
        page = str(source_info.get("page", "Unknown page"))
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        enriched.append(
            {
                "source": source,
                "page": page,
                "nice_ref": _extract_nice_reference(source),
                "excerpt": chunk_lookup.get(key, ""),
            }
        )
    return enriched


def _calculate_confidence(sources):
    """Calculate the requested proxy confidence score from citation coverage."""
    top_k = max(1, int(os.getenv("TOP_K", "5")))
    cited_sources = [
        source
        for source in sources
        if str(source.get("page", "")).strip()
        and str(source.get("page", "")).lower() != "unknown page"
    ]
    return min(100, int((len(cited_sources) / top_k) * 100 + 20))


def _estimate_total_tokens(question, answer, chunks):
    """Estimate total tokens because the backend response does not expose usage."""
    context_text = " ".join(chunk.page_content[:250] for chunk in chunks)
    characters = len(question) + len(answer) + len(context_text)
    return max(1, int(characters / 4))


if __name__ == "__main__":
    main()
