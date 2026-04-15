"""
Multimodal Reranking RAG — Streamlit Frontend
=============================================
Pages
  • Chat         — persistent conversation, animated pipeline progress per query
  • Document Vault — upload PDFs and trigger the ingestion pipeline
"""

import os
import pathlib
import tempfile
import threading
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8080/api/v1")

# Estimated seconds each RAG pipeline step takes (used for progress animation)
# Tweak if your hardware is faster/slower.
_STEP_DURATIONS = [1.0, 4.0, 1.2, 1.0, 3.0, 1.2]   # Route, Retrieve, Rerank, Validate, Generate, Verify

st.set_page_config(
    page_title="Nexus — Multimodal RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }

/* ── Background ─────────────────────────────────────────────────────────── */
.stApp {
    background-color: #0c0c0e;
    background-image:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(212,163,78,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 90% 80%, rgba(212,163,78,0.04) 0%, transparent 60%);
    color: #d4d4d8;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0f0f11 !important;
    border-right: 1px solid #1f1f23 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }

/* ── Hide chrome ─────────────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1080px;
}

/* ── Wordmark ────────────────────────────────────────────────────────────── */
.nx-wordmark {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #f5f5f5;
    letter-spacing: -0.01em;
    margin-bottom: 0.15rem;
}
.nx-wordmark span { color: #d4a34e; }
.nx-tagline {
    font-size: 0.72rem;
    font-weight: 300;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #52525b;
    margin-bottom: 2.4rem;
}

/* ── Sidebar radio nav ───────────────────────────────────────────────────── */
div[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 0.55rem 0.8rem !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    color: #a1a1aa !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    margin-bottom: 0.2rem !important;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(212,163,78,0.07) !important;
    color: #d4a34e !important;
}
div[data-testid="stRadio"] label[aria-checked="true"] {
    background: rgba(212,163,78,0.1) !important;
    color: #d4a34e !important;
}
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { margin: 0 !important; }

/* ── Page hero ───────────────────────────────────────────────────────────── */
.nx-hero-eyebrow {
    font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #d4a34e; margin-bottom: 0.6rem;
}
.nx-hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem; font-weight: 400;
    color: #fafafa; line-height: 1.15;
    letter-spacing: -0.02em; margin-bottom: 0.7rem;
}
.nx-hero-title em { color: #d4a34e; font-style: italic; }
.nx-hero-sub {
    font-size: 0.88rem; font-weight: 300;
    color: #71717a; line-height: 1.6;
    margin-bottom: 1.5rem; max-width: 540px;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
.nx-rule {
    height: 1px;
    background: linear-gradient(90deg, #1f1f23 0%, #d4a34e22 40%, #1f1f23 100%);
    margin: 1.5rem 0; border: none;
}

/* ── Pipeline tracker ────────────────────────────────────────────────────── */
.nx-pipeline {
    display: flex; align-items: center;
    gap: 0; margin-bottom: 1.6rem;
    overflow-x: auto; padding-bottom: 0.2rem;
}
.nx-pip-step {
    display: flex; align-items: center; gap: 0.45rem;
    padding: 0.4rem 0.9rem;
    border: 1px solid #272729; border-radius: 0;
    font-size: 0.73rem; font-weight: 500;
    letter-spacing: 0.04em;
    color: #52525b; white-space: nowrap;
    background: #0f0f11;
    transition: color 0.4s, border-color 0.4s, background 0.4s;
}
.nx-pip-step:first-child { border-radius: 6px 0 0 6px; }
.nx-pip-step:last-child  { border-radius: 0 6px 6px 0; }

/* dot states */
.nx-pip-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #3f3f46; flex-shrink: 0;
    transition: background 0.4s;
}
/* active = pulsing amber */
.nx-pip-step.active {
    color: #d4a34e;
    border-color: #d4a34e55;
    background: rgba(212,163,78,0.06);
}
.nx-pip-step.active .nx-pip-dot {
    background: #d4a34e;
    box-shadow: 0 0 0 3px rgba(212,163,78,0.25);
    animation: pip-pulse 0.9s ease-in-out infinite;
}
/* done = solid green */
.nx-pip-step.done {
    color: #4ade80;
    border-color: #4ade8044;
    background: rgba(74,222,128,0.05);
}
.nx-pip-step.done .nx-pip-dot {
    background: #4ade80;
    box-shadow: none;
}
@keyframes pip-pulse {
    0%, 100% { box-shadow: 0 0 0 3px rgba(212,163,78,0.25); }
    50%       { box-shadow: 0 0 0 6px rgba(212,163,78,0.12); }
}
.nx-pip-arrow {
    width: 24px; height: 1px;
    background: #272729; flex-shrink: 0;
}

/* ── Chat history ────────────────────────────────────────────────────────── */
.nx-chat-turn { margin-bottom: 2rem; }

/* User bubble */
.nx-user-bubble {
    display: flex; justify-content: flex-end;
    margin-bottom: 0.6rem;
}
.nx-user-bubble-inner {
    background: rgba(212,163,78,0.1);
    border: 1px solid rgba(212,163,78,0.22);
    border-radius: 14px 14px 4px 14px;
    padding: 0.7rem 1.1rem;
    max-width: 75%;
    font-size: 0.92rem;
    color: #e4e4e7;
    line-height: 1.6;
}

/* Assistant card */
.nx-answer-wrap {
    background: #111113;
    border: 1px solid #1f1f23;
    border-radius: 4px 14px 14px 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 0.6rem;
    position: relative;
    overflow: hidden;
}
.nx-answer-wrap::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #4ade8055, transparent);
}
.nx-answer-eyebrow {
    font-size: 0.6rem; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #4ade80; margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.nx-answer-eyebrow::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%; background: #4ade80;
}
.nx-answer-body {
    font-size: 0.97rem; font-weight: 300;
    color: #d4d4d8; line-height: 1.8;
}

/* ── Meta row cards ──────────────────────────────────────────────────────── */
.nx-meta-card {
    background: #111113;
    border: 1px solid #1f1f23;
    border-radius: 10px;
    padding: 0.9rem 1.1rem; height: 100%;
}
.nx-meta-label {
    font-size: 0.6rem; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #52525b; margin-bottom: 0.4rem;
}
.nx-meta-value {
    font-size: 0.88rem; font-weight: 400;
    color: #e4e4e7; line-height: 1.5; word-break: break-word;
}
.nx-citation-chip {
    display: inline-block;
    background: rgba(212,163,78,0.1);
    border: 1px solid rgba(212,163,78,0.25);
    border-radius: 5px;
    padding: 0.18rem 0.55rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.73rem; color: #d4a34e;
}

/* ── Image evidence ──────────────────────────────────────────────────────── */
.nx-img-block {
    border: 1px solid #1f1f23; border-radius: 10px;
    overflow: hidden; margin: 0.8rem 0 1rem;
    background: #111113;
}
.nx-img-header {
    padding: 0.6rem 1.1rem; border-bottom: 1px solid #1f1f23;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #52525b; display: flex; align-items: center; gap: 0.5rem;
}
.nx-img-header::before {
    content: ''; display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%; background: #d4a34e;
}
.nx-img-caption {
    padding: 0.4rem 1.1rem 0.6rem;
    font-size: 0.71rem; color: #52525b;
    font-family: 'DM Mono', monospace;
}

/* ── Native chat input (st.chat_input) — pinned bottom ──────────────────── */
/* Target every layer Streamlit wraps around the textarea */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    background: #0c0c0e !important;
}
[data-testid="stBottom"] {
    border-top: 1px solid #1a1a1e !important;
    padding: 0.9rem 0 1.1rem !important;
}

/* The main chat input container */
[data-testid="stChatInputContainer"],
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] > div > div > div {
    background: #111113 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] > div {
    border: 1px solid #272729 !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(212,163,78,0.45) !important;
    box-shadow: 0 0 0 3px rgba(212,163,78,0.07) !important;
}

/* Textarea itself */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    background: #111113 !important;
    color: #e4e4e7 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
    font-weight: 300 !important;
    caret-color: #d4a34e !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: #52525b !important;
}

/* Send button */
[data-testid="stChatInput"] button,
[data-testid="stChatInputContainer"] button {
    background: #d4a34e !important;
    border-radius: 8px !important;
    color: #0c0c0e !important;
    border: none !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"] button:hover,
[data-testid="stChatInputContainer"] button:hover {
    opacity: 0.82 !important;
}
[data-testid="stChatInput"] button svg,
[data-testid="stChatInputContainer"] button svg {
    fill: #0c0c0e !important;
    stroke: #0c0c0e !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: #d4a34e !important;
    border: none !important; border-radius: 8px !important;
    color: #0c0c0e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.58rem 1.4rem !important;
    transition: opacity 0.18s, transform 0.12s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

.nx-ghost-btn button {
    background: transparent !important;
    border: 1px solid #272729 !important;
    color: #71717a !important; font-size: 0.78rem !important;
}
.nx-ghost-btn button:hover {
    border-color: #52525b !important; color: #a1a1aa !important;
    transform: none !important; opacity: 1 !important;
}



/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed #272729 !important; border-radius: 12px !important;
    background: #0f0f11 !important; padding: 1.5rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: #d4a34e66 !important; }
[data-testid="stFileUploader"] section > div { background: transparent !important; }

.nx-file-item {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.6rem 0.9rem; border: 1px solid #1f1f23;
    border-radius: 8px; margin-bottom: 0.5rem; background: #111113;
}
.nx-file-icon { color: #d4a34e; font-size: 0.85rem; }
.nx-file-name { color: #e4e4e7; font-size: 0.85rem; flex: 1; }
.nx-file-size { color: #52525b; font-size: 0.75rem; font-family: 'DM Mono', monospace; }

/* ── Ingestion results ───────────────────────────────────────────────────── */
.nx-result-ok {
    display: flex; align-items: flex-start; gap: 0.8rem;
    padding: 0.85rem 1rem; border: 1px solid #1a2e1a;
    border-radius: 10px; background: #0f1a0f; margin-bottom: 0.5rem;
}
.nx-result-err {
    display: flex; align-items: flex-start; gap: 0.8rem;
    padding: 0.85rem 1rem; border: 1px solid #2e1a1a;
    border-radius: 10px; background: #1a0f0f; margin-bottom: 0.5rem;
}
.nx-result-icon { font-size: 1rem; flex-shrink: 0; margin-top: 0.05rem; }
.nx-result-text { font-size: 0.83rem; color: #d4d4d8; line-height: 1.5; }
.nx-result-text strong { color: #e4e4e7; font-weight: 500; }
.nx-result-text code {
    font-family: 'DM Mono', monospace; font-size: 0.77rem;
    background: rgba(255,255,255,0.05);
    padding: 0.05rem 0.32rem; border-radius: 4px; color: #a1a1aa;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #1f1f23 !important; border-radius: 10px !important;
    background: #0f0f11 !important;
}
[data-testid="stExpander"] summary { color: #71717a !important; font-size: 0.82rem !important; }

/* ── Progress bar ────────────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #d4a34e, #e8c077) !important;
    border-radius: 99px !important;
}
.stProgress > div > div { border-radius: 99px !important; background: #1f1f23 !important; }

/* ── Alert / spinner ─────────────────────────────────────────────────────── */
.stAlert { border-radius: 10px !important; font-size: 0.84rem !important; }
.stSpinner > div { border-top-color: #d4a34e !important; }

/* ── Table ───────────────────────────────────────────────────────────────── */
table { border-collapse: collapse !important; width: 100% !important; font-size: 0.83rem !important; }
th {
    text-align: left !important; padding: 0.5rem 0.8rem !important;
    border-bottom: 1px solid #272729 !important; color: #71717a !important;
    font-weight: 500 !important; letter-spacing: 0.05em !important;
}
td {
    padding: 0.52rem 0.8rem !important; border-bottom: 1px solid #1a1a1e !important;
    color: #d4d4d8 !important; vertical-align: top !important;
}
tr:last-child td { border-bottom: none !important; }

/* ── Tech chips ──────────────────────────────────────────────────────────── */
.nx-tech-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.8rem; }
.nx-tech-chip {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    color: #52525b; border: 1px solid #1f1f23;
    border-radius: 4px; padding: 0.15rem 0.45rem;
}

/* ── Typing indicator ────────────────────────────────────────────────────── */
.nx-typing {
    display: flex; align-items: center; gap: 5px;
    padding: 0.7rem 1rem;
    background: #111113; border: 1px solid #1f1f23;
    border-radius: 4px 14px 14px 14px;
    width: fit-content; margin-bottom: 0.6rem;
}
.nx-typing span {
    width: 7px; height: 7px; border-radius: 50%;
    background: #3f3f46; display: inline-block;
    animation: jump 1.2s ease-in-out infinite;
}
.nx-typing span:nth-child(2) { animation-delay: 0.2s; }
.nx-typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes jump {
    0%, 80%, 100% { transform: translateY(0); background: #3f3f46; }
    40% { transform: translateY(-6px); background: #d4a34e; }
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #272729; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="nx-wordmark">Ne<span>x</span>us</div>'
        '<div class="nx-tagline">Multimodal Intelligence</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        ["Chat", "Document Vault"],
        label_visibility="collapsed",
        format_func=lambda x: f"{'◈  ' if x == 'Chat' else '⊞  '}{x}",
    )

    st.markdown("<div class='nx-rule'></div>", unsafe_allow_html=True)

    if page == "Chat" and st.button("Clear conversation", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

    


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_STEPS = ["Route", "Retrieve", "Rerank", "Validate", "Generate", "Verify"]


def _post_query(query: str) -> dict:
    resp = requests.post(f"{API_BASE}/query/", json={"query": query}, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _pipeline_html(active: int = -1, done_up_to: int = -1) -> str:
    """
    Render the pipeline strip as HTML.
      active      = index of the currently running step (-1 = none)
      done_up_to  = all steps with index < done_up_to are marked done
    """
    parts = []
    for i, label in enumerate(PIPELINE_STEPS):
        if i < done_up_to:
            cls = "nx-pip-step done"
            dot = '<span class="nx-pip-dot"></span>'
            # replace dot with a tiny checkmark feel
        elif i == active:
            cls = "nx-pip-step active"
            dot = '<span class="nx-pip-dot"></span>'
        else:
            cls = "nx-pip-step"
            dot = '<span class="nx-pip-dot"></span>'

        parts.append(f'<div class="{cls}">{dot}{label}</div>')
        if i < len(PIPELINE_STEPS) - 1:
            parts.append('<div class="nx-pip-arrow"></div>')

    return f'<div class="nx-pipeline">{"".join(parts)}</div>'


def _run_with_pipeline_animation(query: str):
    """
    Fires the API call in a background thread while animating the
    pipeline steps live with green progress.
    Returns (result_dict | None, error_str | None).
    """
    result_holder: list = [None]
    error_holder:  list = [None]

    def _fetch():
        try:
            result_holder[0] = _post_query(query)
        except Exception as exc:
            error_holder[0] = exc

    thread = threading.Thread(target=_fetch, daemon=True)
    thread.start()

    # ── Live pipeline animation ───────────────────────────────────────────
    pip_slot    = st.empty()
    typing_slot = st.empty()

    total_steps = len(PIPELINE_STEPS)
    step_idx    = 0
    done_count  = 0

    typing_slot.markdown(
        '<div class="nx-typing"><span></span><span></span><span></span></div>',
        unsafe_allow_html=True,
    )

    while thread.is_alive() or step_idx < total_steps:
        pip_slot.markdown(_pipeline_html(active=step_idx, done_up_to=done_count), unsafe_allow_html=True)

        # Wait for this step's estimated duration, then advance
        deadline = time.time() + _STEP_DURATIONS[step_idx]
        while time.time() < deadline:
            if not thread.is_alive():
                # API finished early — fast-forward remaining steps
                break
            time.sleep(0.08)

        done_count = step_idx + 1
        step_idx   = min(step_idx + 1, total_steps - 1)

        # If API finished and we've shown at least the Generate step, stop looping
        if not thread.is_alive() and done_count >= total_steps:
            break

        # If API finished mid-loop, finish remaining steps quickly
        if not thread.is_alive():
            for remaining in range(step_idx, total_steps):
                pip_slot.markdown(_pipeline_html(active=remaining, done_up_to=remaining), unsafe_allow_html=True)
                time.sleep(0.18)
                done_count = remaining + 1
            break

    # Final state: all green
    pip_slot.markdown(_pipeline_html(active=-1, done_up_to=total_steps), unsafe_allow_html=True)
    typing_slot.empty()

    thread.join(timeout=5)
    return result_holder[0], error_holder[0]


def _render_answer(result: dict) -> None:
    """Render a single structured RAG response (used in chat history)."""
    answer  = result.get("answer", "—")
    img_path = result.get("image_path")
    citation = result.get("policy_citations", "—")
    doc_name = result.get("document_name", "—")
    page_no  = result.get("page_no", "—")

    st.markdown(
        f'<div class="nx-answer-wrap">'
        f'<div class="nx-answer-eyebrow">Nexus</div>'
        f'<div class="nx-answer-body">{answer}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if img_path and pathlib.Path(img_path).exists():
        st.markdown(
            '<div class="nx-img-block"><div class="nx-img-header">Visual Evidence</div>',
            unsafe_allow_html=True,
        )
        img_col, _ = st.columns([2, 1])
        with img_col:
            st.image(img_path, use_container_width=True)
        st.markdown(
            f'<div class="nx-img-caption">{doc_name} · p.{page_no}</div></div>',
            unsafe_allow_html=True,
        )
    elif img_path:
        st.warning(f"Image not found: `{img_path}`")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="nx-meta-card"><div class="nx-meta-label">Document</div>'
            f'<div class="nx-meta-value">{doc_name}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="nx-meta-card"><div class="nx-meta-label">Page</div>'
            f'<div class="nx-meta-value">{page_no}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="nx-meta-card"><div class="nx-meta-label">Citation</div>'
            f'<div class="nx-meta-value"><span class="nx-citation-chip">{citation}</span></div></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Chat
# ─────────────────────────────────────────────────────────────────────────────
if page == "Chat":

    # Compact header
    st.markdown(
        '<div class="nx-hero-eyebrow">Intelligence layer</div>'
        '<div class="nx-hero-title">Ask anything.<br><em>Get precise answers.</em></div>'
        '<div class="nx-hero-sub">'
        'Hybrid retrieval · Cohere reranking · Hallucination-verified answers'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Session init ──────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Render full chat history ───────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="nx-user-bubble">'
                f'<div class="nx-user-bubble-inner">{msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            # Render the full pipeline strip (all done) for completed turns
            st.markdown(
                _pipeline_html(active=-1, done_up_to=len(PIPELINE_STEPS)),
                unsafe_allow_html=True,
            )
            _render_answer(msg["result"])
            with st.expander("Raw JSON"):
                st.json(msg["result"])


    # ── Pending user query detection ──────────────────────────────────────────
    # True when the last message is from the user and still has no assistant reply
    needs_reply = (
        bool(st.session_state.messages)
        and st.session_state.messages[-1]["role"] == "user"
    )

    # ── Resolve pending query (runs BEFORE chat_input so animation shows inline)
    if needs_reply:
        user_q = st.session_state.messages[-1]["content"]

        # User bubble for this in-progress turn
        st.markdown(
            f'<div class="nx-user-bubble">'
            f'<div class="nx-user-bubble-inner">{user_q}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        result, error = _run_with_pipeline_animation(user_q)

        if error:
            if isinstance(error, requests.HTTPError):
                st.error(f"API {error.response.status_code} — {error.response.text}")
            elif isinstance(error, requests.ConnectionError):
                st.error(
                    f"Cannot reach API at `{API_BASE}`. "
                    "Make sure the FastAPI server is running."
                )
            else:
                st.error(str(error))
            st.session_state.messages.pop()   # let user retry
        else:
            st.session_state.messages.append({"role": "assistant", "result": result})
            st.rerun()

    # ── Native sticky bottom chat input ──────────────────────────────────────
    # st.chat_input() is automatically pinned to the bottom of the viewport
    if user_prompt := st.chat_input("Ask a question about your documents…"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page: Document Vault
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Document Vault":

    st.markdown(
        '<div class="nx-hero-eyebrow">Knowledge base</div>'
        '<div class="nx-hero-title">Document <em>Vault.</em></div>'
        '<div class="nx-hero-sub">'
        'Upload PDFs — Docling parses text, tables and images into semantically '
        'rich chunks, embedded and indexed in pgvector for instant retrieval.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Static ingest pipeline indicator
    ingest_steps = ["Upload", "Parse", "Chunk", "Embed", "Index"]
    parts = []
    for i, label in enumerate(ingest_steps):
        parts.append(f'<div class="nx-pip-step"><span class="nx-pip-dot"></span>{label}</div>')
        if i < len(ingest_steps) - 1:
            parts.append('<div class="nx-pip-arrow"></div>')
    st.markdown(f'<div class="nx-pipeline">{"".join(parts)}</div>', unsafe_allow_html=True)

    st.markdown("<div class='nx-rule'></div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "drop",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(
            f"<div style='font-size:0.72rem;color:#52525b;letter-spacing:0.1em;"
            f"text-transform:uppercase;margin:1rem 0 0.6rem;font-weight:600;'>"
            f"{len(uploaded_files)} file{'s' if len(uploaded_files)>1 else ''} ready</div>",
            unsafe_allow_html=True,
        )
        for f in uploaded_files:
            size_kb = round(f.size / 1024, 1)
            st.markdown(
                f'<div class="nx-file-item">'
                f'<span class="nx-file-icon">⊞</span>'
                f'<span class="nx-file-name">{f.name}</span>'
                f'<span class="nx-file-size">{size_kb} KB</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    ingest_btn = st.button("Ingest Documents", disabled=not uploaded_files)

    if ingest_btn and uploaded_files:
        results = []
        progress = st.progress(0, text="Starting…")

        for idx, uploaded_file in enumerate(uploaded_files):
            progress.progress(idx / len(uploaded_files), text=f"Processing {uploaded_file.name}…")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="rag_ingest_") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                from src.ingestion.ingestion import run_ingestion
                result = run_ingestion(tmp_path)
                results.append({
                    "file":   uploaded_file.name,
                    "status": result.get("status", "success"),
                    "doc_id": result.get("doc_id", "—"),
                    "chunks": result.get("chunks_ingested", 0),
                })
            except Exception as exc:
                results.append({"file": uploaded_file.name, "status": "error", "error": str(exc)})
            finally:
                pathlib.Path(tmp_path).unlink(missing_ok=True)

        progress.progress(1.0, text="Complete")
        st.markdown("<div class='nx-rule'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.68rem;font-weight:600;letter-spacing:0.14em;"
            "text-transform:uppercase;color:#52525b;margin-bottom:0.8rem;'>Results</div>",
            unsafe_allow_html=True,
        )
        for r in results:
            if r.get("status") == "success":
                st.markdown(
                    f'<div class="nx-result-ok">'
                    f'<span class="nx-result-icon">✓</span>'
                    f'<div class="nx-result-text">'
                    f'<strong>{r["file"]}</strong> — {r["chunks"]} chunks stored &nbsp;'
                    f'<code>{r["doc_id"]}</code>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="nx-result-err">'
                    f'<span class="nx-result-icon">✕</span>'
                    f'<div class="nx-result-text">'
                    f'<strong>{r["file"]}</strong> — {r.get("error", "Unknown error")}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<div class='nx-rule'></div>", unsafe_allow_html=True)
    with st.expander("What happens during ingestion?"):
        st.markdown("""
| Step | Process |
|------|---------|
| **Parse** | Docling extracts text, tables and images with layout awareness |
| **Chunk** | Long blocks split into 1500-char windows with 300-char overlap |
| **Embed** | Every chunk embedded with `gemini-embedding-2-preview` (1536 dims) |
| **Index** | Vectors + metadata written to `multimodal_chunks` in pgvector |

Re-uploading the same filename is **idempotent** — old chunks are replaced, not duplicated.
        """)
