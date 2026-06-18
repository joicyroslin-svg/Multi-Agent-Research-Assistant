import hashlib
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from utils.agents import (
    research_agent,
    summary_agent,
    notes_agent,
    question_agent,
    report_agent
)
from utils.agent_planner import planner_agent
from utils.document_reader import (
    extract_text_from_pdf,
    extract_text_from_txt,
    split_text_into_chunks
)
from utils.vector_store import (
    combine_vector_chunks,
    index_document_chunks,
    query_vector_database
)


st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="MA",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #f4f7fb;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #dbe4ef;
    --blue: #2563eb;
    --blue-dark: #1e40af;
    --teal: #0f766e;
    --green: #16a34a;
    --amber: #b45309;
    --violet: #6d28d9;
    --shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 12% 0%, rgba(37, 99, 235, 0.08), transparent 28%),
        radial-gradient(circle at 88% 8%, rgba(15, 118, 110, 0.10), transparent 26%),
        var(--bg);
}

.block-container {
    max-width: 1440px;
    padding-top: 1.3rem;
    padding-bottom: 2.6rem;
}

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 8% 0%, rgba(37, 99, 235, 0.13), transparent 30%),
        radial-gradient(circle at 92% 10%, rgba(20, 184, 166, 0.16), transparent 28%),
        #f8fbff;
    border-right: 1px solid #dbe4ef;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1.4rem 1rem;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small {
    color: #344054 !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    color: inherit !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"],
section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    padding: 12px;
    color: #344054;
}

section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    min-height: 44px;
    color: #0f172a !important;
    caret-color: #2563eb;
    font-weight: 700;
}

section[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
    color: #94a3b8 !important;
}

section[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: #38bdf8;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.18);
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #2563eb, #0f766e);
    color: #ffffff !important;
    border-radius: 10px;
    border: 0;
    font-weight: 800;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #1d4ed8, #0d9488);
    color: #ffffff !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] div,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    color: #344054 !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #667085 !important;
}

section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"],
section[data-testid="stSidebar"] [data-testid="stSlider"] div {
    color: #344054 !important;
}

section[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(135deg, #2563eb, #14b8a6);
    border: 0;
    border-radius: 12px;
    color: white;
    font-weight: 800;
    min-height: 44px;
}

section[data-testid="stSidebar"] .stButton>button:hover {
    background: linear-gradient(135deg, #7c3aed, #0891b2);
    color: #ffffff;
}

.sidebar-brand {
    padding: 6px 0 16px 0;
    border-bottom: 1px solid #dbe4ef;
    margin-bottom: 18px;
}

.sidebar-logo {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    background: conic-gradient(from 180deg, #14b8a6, #38bdf8, #7c3aed, #f97316, #14b8a6);
    color: white;
    font-weight: 800;
    margin-bottom: 12px;
}

.sidebar-title {
    font-size: 18px;
    font-weight: 800;
    line-height: 1.15;
    color: #0f172a;
}

.sidebar-copy {
    color: #667085;
    font-size: 13px;
    line-height: 1.45;
    margin-top: 8px;
}

.sidebar-card {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 14px;
    padding: 14px;
    margin: 14px 0;
    box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
}

.sidebar-card-title {
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    color: #2563eb;
    margin-bottom: 10px;
}

.sidebar-stat {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 0;
    border-top: 1px solid #e6edf6;
    font-size: 13px;
    color: #344054;
}

.sidebar-stat span {
    color: #667085 !important;
}

.sidebar-stat b {
    color: #0f172a !important;
}

.sidebar-stat:first-of-type {
    border-top: 0;
}

.app-shell {
    display: grid;
    gap: 18px;
}

.hero {
    background:
        radial-gradient(circle at 85% 10%, rgba(249, 115, 22, 0.28), transparent 25%),
        linear-gradient(135deg, rgba(15, 23, 42, 0.97), rgba(49, 46, 129, 0.90)),
        linear-gradient(90deg, rgba(20,184,166,0.38), rgba(37,99,235,0.18));
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 22px;
    padding: 26px;
    color: white;
    box-shadow: 0 28px 70px rgba(15, 23, 42, 0.20);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    inset: auto 0 0 0;
    height: 5px;
    background: linear-gradient(90deg, #14b8a6, #38bdf8, #2563eb, #7c3aed, #f97316);
}

.hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.32fr) minmax(280px, 0.68fr);
    gap: 24px;
    align-items: center;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 0 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    color: #dbeafe;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 42px;
    line-height: 1.04;
    font-weight: 800;
    max-width: 780px;
}

.hero-copy {
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.65;
    max-width: 800px;
    margin-top: 12px;
}

.confidence-card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 18px;
    padding: 18px;
    backdrop-filter: blur(12px);
}

.confidence-label {
    color: #bfdbfe;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}

.confidence-value {
    font-size: 54px;
    font-weight: 800;
    line-height: 1;
    margin-top: 12px;
}

.confidence-note {
    color: #dbeafe;
    font-size: 13px;
    margin-top: 10px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
}

.metric {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 18px;
    box-shadow: var(--shadow);
}

.metric-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}

.metric-token {
    width: 30px;
    height: 30px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #eff6ff, #ecfeff);
    color: #075985;
    font-size: 11px;
    font-weight: 800;
}

.metric-value {
    color: var(--ink);
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
    margin-top: 12px;
}

.metric-note {
    color: var(--muted);
    font-size: 12px;
    margin-top: 9px;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(330px, 0.48fr);
    gap: 18px;
    align-items: start;
}

.panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: var(--shadow);
    padding: 18px;
}

.panel-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 14px;
}

.panel-title {
    color: var(--ink);
    font-size: 18px;
    font-weight: 800;
}

.panel-copy {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
    margin-top: 4px;
}

.pill {
    display: inline-flex;
    align-items: center;
    min-height: 30px;
    padding: 0 11px;
    border-radius: 999px;
    color: #344054;
    background: #f8fafc;
    border: 1px solid var(--line);
    font-size: 12px;
    font-weight: 800;
}

.pill-good {
    color: #047857;
    background: #ecfdf5;
    border-color: #bbf7d0;
}

.pill-warn {
    color: #92400e;
    background: #fffbeb;
    border-color: #fde68a;
}

.workflow-line {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
}

.agent-step {
    background: #f8fafc;
    border: 1px solid #e6edf6;
    border-radius: 16px;
    padding: 14px;
    min-height: 136px;
    position: relative;
}

.agent-step::after {
    content: "";
    position: absolute;
    top: 31px;
    right: -12px;
    width: 12px;
    height: 2px;
    background: #cbd5e1;
}

.agent-step:last-child::after {
    display: none;
}

.step-index {
    width: 32px;
    height: 32px;
    border-radius: 11px;
    display: grid;
    place-items: center;
    background: #e0f2fe;
    color: #075985;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 12px;
}

.step-title {
    color: var(--ink);
    font-size: 13px;
    font-weight: 800;
    line-height: 1.3;
    min-height: 34px;
}

.status {
    display: inline-block;
    margin-top: 12px;
    border-radius: 999px;
    padding: 6px 9px;
    font-size: 11px;
    font-weight: 800;
}

.status-waiting {
    color: #475467;
    background: #eef2f6;
}

.status-running {
    color: #075985;
    background: #cffafe;
}

.status-completed {
    color: #047857;
    background: #d1fae5;
}

.insight-stack {
    display: grid;
    gap: 14px;
}

.insight {
    background: #f8fafc;
    border: 1px solid #e6edf6;
    border-radius: 16px;
    padding: 14px;
}

.insight-title-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    margin-bottom: 10px;
}

.insight-title {
    color: var(--ink);
    font-size: 14px;
    font-weight: 800;
}

.insight-copy {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
}

.source-list {
    display: grid;
    gap: 8px;
    margin-top: 12px;
}

.source-item {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    border-top: 1px solid #e6edf6;
    padding-top: 8px;
    color: var(--muted);
    font-size: 12px;
}

.output-shell {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: var(--shadow);
    padding: 18px;
}

.planner-box {
    background: #f8fafc;
    border: 1px solid #e6edf6;
    border-left: 4px solid var(--violet);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 12px;
}

.memory-box {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px;
    box-shadow: var(--shadow);
    margin-bottom: 10px;
}

.memory-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
    margin-top: 10px;
}

.memory-stat {
    background: #f8fafc;
    border: 1px solid #e6edf6;
    border-radius: 12px;
    padding: 9px;
    color: var(--muted);
    font-size: 12px;
}

.memory-stat b {
    display: block;
    color: var(--ink);
    font-size: 13px;
    margin-bottom: 2px;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    border: 0;
    min-height: 44px;
    background: linear-gradient(135deg, #2563eb, #14b8a6);
    color: #ffffff;
    font-weight: 800;
    box-shadow: 0 14px 30px rgba(37, 99, 235, 0.20);
}

.stButton>button:hover {
    color: #ffffff;
    background: linear-gradient(135deg, #7c3aed, #0891b2);
}

.stDownloadButton>button {
    width: 100%;
    border-radius: 12px;
    min-height: 42px;
    font-weight: 800;
}

[data-testid="stProgress"] > div {
    border-radius: 999px;
    background-color: #dbe4ee;
}

[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0f766e, #2563eb);
}

[data-testid="stTabs"] {
    background: #f8fafc;
    border: 1px solid #e6edf6;
    border-radius: 16px;
    padding: 6px 6px 14px 6px;
}

[data-testid="stTabs"] button {
    border-radius: 11px 11px 0 0;
    font-weight: 750;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    color: var(--ink);
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

@media (max-width: 1050px) {
    .hero-grid,
    .metric-grid,
    .dashboard-grid,
    .workflow-line,
    .memory-grid {
        grid-template-columns: 1fr;
    }

    .agent-step::after {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "research": "",
        "summary": "",
        "notes": "",
        "questions": "",
        "report": "",
        "document_text": "",
        "retrieved_context": "",
        "retrieved_chunks": [],
        "indexed_chunks": 0,
        "indexed_document_hash": "",
        "planner": {},
        "memory": [],
        "history": [],
        "workflow_status": {
            "Planner Agent": "Waiting",
            "Vector RAG Agent": "Waiting",
            "Research Agent": "Waiting",
            "Summary Agent": "Waiting",
            "Report Agent": "Waiting"
        }
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_outputs():
    st.session_state.research = ""
    st.session_state.summary = ""
    st.session_state.notes = ""
    st.session_state.questions = ""
    st.session_state.report = ""
    st.session_state.retrieved_context = ""
    st.session_state.retrieved_chunks = []
    st.session_state.planner = {}
    st.session_state.workflow_status = {
        "Planner Agent": "Waiting",
        "Vector RAG Agent": "Waiting",
        "Research Agent": "Waiting",
        "Summary Agent": "Waiting",
        "Report Agent": "Waiting"
    }


def status_class(status):
    return f"status-{status.lower()}"


def show_agent_step(index, title, agent_name):
    status = st.session_state.workflow_status[agent_name]

    st.markdown(
        f"""
        <div class="agent-step">
            <div class="step-index">{index}</div>
            <div class="step-title">{title}</div>
            <div class="status {status_class(status)}">{status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def count_completed_agents():
    return sum(
        1 for value in st.session_state.workflow_status.values()
        if value == "Completed"
    )


def rag_confidence():
    if not st.session_state.retrieved_chunks:
        return 0

    total = sum(item["score"] for item in st.session_state.retrieved_chunks)
    return round(total / len(st.session_state.retrieved_chunks), 3)


def confidence_label(confidence):
    if confidence >= 0.7:
        return "High confidence", "pill-good"
    if confidence > 0:
        return "Needs review", "pill-warn"
    return "No evidence yet", ""


def add_memory(topic):
    item = {
        "topic": topic,
        "mode": st.session_state.planner.get("mode", "Unknown"),
        "trust": st.session_state.planner.get("trust_level", "Unknown"),
        "sources": len(st.session_state.retrieved_chunks),
        "confidence": rag_confidence()
    }

    st.session_state.memory.insert(0, item)
    st.session_state.memory = st.session_state.memory[:6]


init_session_state()

completed = count_completed_agents()
total_agents = 5
confidence = rag_confidence()
source_label, source_class = confidence_label(confidence)
current_mode = st.session_state.planner.get("mode", "Ready")

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">MA</div>
            <div class="sidebar-title">Multi-Agent Research Assistant</div>
                    <div class="sidebar-copy">Create research, notes, questions, and reports from a topic or uploaded document.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    topic = st.text_input(
        "Research Topic",
        placeholder="Example: Retrieval Augmented Generation"
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT document",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        if uploaded_file.name.lower().endswith(".pdf"):
            st.session_state.document_text = extract_text_from_pdf(uploaded_file)
        else:
            st.session_state.document_text = extract_text_from_txt(uploaded_file)

        st.success("Document extracted successfully.")
        st.caption(f"Extracted words: {len(st.session_state.document_text.split())}")

    top_k = st.slider(
        "Document Sections to Use",
        2,
        6,
        4,
        help="How many relevant document sections the app should use as evidence."
    )

    run_button = st.button("Run Multi-Agent Research")
    clear_button = st.button("Clear Workspace")

    if clear_button:
        reset_outputs()
        st.session_state.document_text = ""
        st.session_state.indexed_chunks = 0
        st.session_state.indexed_document_hash = ""
        st.success("Workspace cleared.")

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Current Session</div>
            <div class="sidebar-stat"><span>Mode</span><b>{current_mode}</b></div>
            <div class="sidebar-stat"><span>Evidence found</span><b>{len(st.session_state.retrieved_chunks)}</b></div>
            <div class="sidebar-stat"><span>Topics</span><b>{len(st.session_state.history)}</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown('<div class="app-shell">', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="eyebrow">Research Dashboard</div>
                <div class="hero-title">Multi-Agent Research Assistant</div>
                <div class="hero-copy">
                    Upload a document, find relevant evidence, let the planner choose the workflow,
                    and generate research, notes, questions, and a final report from one focused dashboard.
                </div>
            </div>
            <div class="confidence-card">
                <div class="confidence-label">RAG Confidence</div>
                <div class="confidence-value">{confidence}</div>
                <div class="confidence-note">{source_label if confidence else "Run with a document to calculate confidence"}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="metric-grid">
        <div class="metric">
            <div class="metric-top">
                <div class="metric-label">Progress</div>
                <div class="metric-token">P</div>
            </div>
            <div class="metric-value">{completed}/{total_agents}</div>
            <div class="metric-note">completed workflow stages</div>
        </div>
        <div class="metric">
            <div class="metric-top">
                <div class="metric-label">Indexed Evidence</div>
                <div class="metric-token">DB</div>
            </div>
            <div class="metric-value">{st.session_state.indexed_chunks}</div>
            <div class="metric-note">document sections indexed</div>
        </div>
        <div class="metric">
            <div class="metric-top">
                <div class="metric-label">Evidence</div>
                <div class="metric-token">S</div>
            </div>
            <div class="metric-value">{len(st.session_state.retrieved_chunks)}</div>
            <div class="metric-note">document sections found</div>
        </div>
        <div class="metric">
            <div class="metric-top">
                <div class="metric-label">Memory</div>
                <div class="metric-token">M</div>
            </div>
            <div class="metric-value">{len(st.session_state.memory)}</div>
            <div class="metric-note">recent research runs</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.progress(completed / total_agents)

left, right = st.columns([1, 0.48], gap="large")

with left:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-head">
                <div>
                    <div class="panel-title">Workflow</div>
                    <div class="panel-copy">Five agents collaborate from planning to final report generation.</div>
                </div>
                <span class="pill">Pipeline</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    workflow_cols = st.columns(5)
    with workflow_cols[0]:
        show_agent_step("01", "Find document evidence", "Vector RAG Agent")
    with workflow_cols[1]:
        show_agent_step("02", "Plan research mode", "Planner Agent")
    with workflow_cols[2]:
        show_agent_step("03", "Generate research", "Research Agent")
    with workflow_cols[3]:
        show_agent_step("04", "Create study aids", "Summary Agent")
    with workflow_cols[4]:
        show_agent_step("05", "Prepare report", "Report Agent")

with right:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-head">
                <div>
                    <div class="panel-title">Planner</div>
                    <div class="panel-copy">Autonomous routing and trust level.</div>
                </div>
                <span class="pill">{st.session_state.planner.get("trust_level", "Waiting")}</span>
            </div>
            <div class="insight-copy">{st.session_state.planner.get("reason", "Run a topic to evaluate document grounding and select the research mode.")}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-head">
                <div>
                    <div class="panel-title">Evidence</div>
                    <div class="panel-copy">Document sections used to ground the answer.</div>
                </div>
                <span class="pill {source_class}">{source_label}</span>
            </div>
            <div class="source-list">
                <div class="source-item"><span>Evidence sections</span><b>{len(st.session_state.retrieved_chunks)}</b></div>
                <div class="source-item"><span>Indexed sections</span><b>{st.session_state.indexed_chunks}</b></div>
                <div class="source-item"><span>Confidence</span><b>{confidence}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if run_button:
    if not topic.strip():
        st.error("Please enter a research topic.")

    else:
        reset_outputs()

        if topic not in st.session_state.history:
            st.session_state.history.append(topic)

        has_document = bool(st.session_state.document_text.strip())

        st.session_state.workflow_status["Vector RAG Agent"] = "Running"

        if has_document:
            document_hash = hashlib.sha256(
                st.session_state.document_text.encode("utf-8", errors="ignore")
            ).hexdigest()

            chunks = split_text_into_chunks(
                st.session_state.document_text,
                chunk_size=1000
            )

            if (
                st.session_state.indexed_document_hash != document_hash
                or st.session_state.indexed_chunks == 0
            ):
                with st.spinner("Building document evidence database..."):
                    st.session_state.indexed_chunks = index_document_chunks(chunks)
                    st.session_state.indexed_document_hash = document_hash

            with st.spinner("Finding relevant document evidence..."):
                st.session_state.retrieved_chunks = query_vector_database(
                    topic,
                    top_k=top_k
                )

            st.session_state.retrieved_context = combine_vector_chunks(
                st.session_state.retrieved_chunks
            )

        else:
            st.session_state.retrieved_context = "No document uploaded. Agents used topic-only GenAI."

        st.session_state.workflow_status["Vector RAG Agent"] = "Completed"

        st.session_state.workflow_status["Planner Agent"] = "Running"

        with st.spinner("Planner Agent is deciding workflow..."):
            st.session_state.planner = planner_agent(
                topic,
                has_document,
                st.session_state.retrieved_chunks
            )

        st.session_state.workflow_status["Planner Agent"] = "Completed"

        st.session_state.workflow_status["Research Agent"] = "Running"

        with st.spinner("Research Agent is creating source-grounded explanation..."):
            st.session_state.research = research_agent(
                topic,
                st.session_state.retrieved_context,
                st.session_state.planner
            )

        st.session_state.workflow_status["Research Agent"] = "Completed"

        st.session_state.workflow_status["Summary Agent"] = "Running"

        with st.spinner("Summary Agent is preparing summary, notes, and questions..."):
            with ThreadPoolExecutor(max_workers=3) as executor:
                summary_future = executor.submit(summary_agent, st.session_state.research)
                notes_future = executor.submit(notes_agent, st.session_state.research)
                questions_future = executor.submit(question_agent, st.session_state.research)

                st.session_state.summary = summary_future.result()
                st.session_state.notes = notes_future.result()
                st.session_state.questions = questions_future.result()

        st.session_state.workflow_status["Summary Agent"] = "Completed"

        st.session_state.workflow_status["Report Agent"] = "Running"

        with st.spinner("Report Agent is building final report..."):
            st.session_state.report = report_agent(
                topic,
                st.session_state.research,
                st.session_state.summary,
                st.session_state.notes,
                st.session_state.questions,
                st.session_state.retrieved_context,
                st.session_state.planner
            )

        st.session_state.workflow_status["Report Agent"] = "Completed"

        add_memory(topic)

        st.success("Multi-agent research completed successfully.")
        st.rerun()


st.markdown("---")

st.markdown('<div class="output-shell">', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Planner",
    "Evidence",
    "Research",
    "Summary",
    "Notes",
    "Questions",
    "Report"
])

with tab1:
    st.markdown("### Planner Output")

    if st.session_state.planner:
        st.markdown(
            f"""
            <div class="planner-box">
                <b>Mode:</b> {st.session_state.planner.get("mode")}<br>
                <b>Trust Level:</b> {st.session_state.planner.get("trust_level")}<br>
                <b>Reason:</b> {st.session_state.planner.get("reason")}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Planned Steps")

        for step in st.session_state.planner.get("steps", []):
            st.write(f"- {step}")

    else:
        st.info("Planner output will appear here.")


with tab2:
    st.markdown("### Retrieved Evidence")

    if st.session_state.retrieved_chunks:
        for item in st.session_state.retrieved_chunks:
            with st.expander(f"{item['source']} | Similarity Score: {item['score']}"):
                st.write(item["chunk"])

    elif st.session_state.retrieved_context:
        st.info(st.session_state.retrieved_context)

    else:
        st.info("Document evidence will appear here.")


with tab3:
    st.markdown("### Research Output")

    if st.session_state.research:
        st.markdown(st.session_state.research)
    else:
        st.info("Research output will appear here.")


with tab4:
    st.markdown("### Summary Output")

    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.info("Summary output will appear here.")


with tab5:
    st.markdown("### Notes Output")

    if st.session_state.notes:
        st.markdown(st.session_state.notes)
    else:
        st.info("Notes output will appear here.")


with tab6:
    st.markdown("### Question Output")

    if st.session_state.questions:
        st.markdown(st.session_state.questions)
    else:
        st.info("Questions will appear here.")


with tab7:
    st.markdown("### Final Report")

    if st.session_state.report:
        st.markdown(st.session_state.report)

        st.download_button(
            label="Download Research Report",
            data=st.session_state.report,
            file_name="multi_agent_research_report.txt",
            mime="text/plain"
        )

    else:
        st.info("Final report will appear here.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("## Research Memory")

if st.session_state.memory:
    for item in st.session_state.memory:
        st.markdown(
            f"""
            <div class="memory-box">
                <b>{item["topic"]}</b>
                <div class="memory-grid">
                    <div class="memory-stat"><b>{item["mode"]}</b>Research Mode</div>
                    <div class="memory-stat"><b>{item["trust"]}</b>Trust Level</div>
                    <div class="memory-stat"><b>{item["sources"]}</b>Evidence Used</div>
                    <div class="memory-stat"><b>{item["confidence"]}</b>RAG Confidence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("Research memory will appear after running a topic.")

st.markdown('</div>', unsafe_allow_html=True)
