import streamlit as st

from utils.agents import (
    research_agent,
    summary_agent,
    notes_agent,
    question_agent,
    report_agent
)

from utils.document_reader import (
    extract_text_from_pdf,
    extract_text_from_txt,
    split_text_into_chunks
)

from utils.vector_store import (
    index_document_chunks,
    query_vector_database,
    combine_vector_chunks
)

from utils.agent_planner import planner_agent


st.set_page_config(
    page_title="Agentic RAG Research SaaS",
    page_icon="🧠",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59, 130, 246, 0.18), transparent 30%),
        radial-gradient(circle at 90% 0%, rgba(124, 58, 237, 0.18), transparent 30%),
        linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #fdf2f8 100%);
}

.block-container {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 45%, #312e81 100%);
    color: white;
    padding: 34px;
    border-radius: 30px;
    box-shadow: 0 30px 70px rgba(15, 23, 42, 0.28);
    margin-bottom: 28px;
}

.badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.13);
    border: 1px solid rgba(255, 255, 255, 0.22);
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 14px;
}

.title {
    font-size: 50px;
    font-weight: 900;
    margin-bottom: 10px;
    letter-spacing: -1.2px;
}

.subtitle {
    font-size: 18px;
    color: #dbeafe;
    max-width: 980px;
    line-height: 1.6;
}

.saas-card {
    background: rgba(255, 255, 255, 0.90);
    backdrop-filter: blur(18px);
    padding: 24px;
    border-radius: 26px;
    border: 1px solid rgba(226, 232, 240, 0.9);
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    margin-bottom: 22px;
}

.metric-card {
    background: linear-gradient(180deg, #ffffff, #f8fafc);
    padding: 22px;
    border-radius: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
}

.metric-value {
    font-size: 34px;
    font-weight: 900;
    color: #2563eb;
}

.metric-label {
    font-size: 14px;
    color: #64748b;
    font-weight: 700;
}

.agent-card {
    background: linear-gradient(180deg, #ffffff, #f8fafc);
    padding: 20px;
    border-radius: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.07);
    text-align: center;
    min-height: 150px;
}

.agent-icon {
    font-size: 30px;
}

.agent-title {
    font-size: 15px;
    font-weight: 900;
    color: #111827;
    margin-top: 8px;
}

.agent-status {
    margin-top: 10px;
    font-size: 13px;
    font-weight: 900;
    color: #4f46e5;
    background: #e0e7ff;
    padding: 7px 12px;
    border-radius: 999px;
    display: inline-block;
}

.section-title {
    font-size: 25px;
    font-weight: 900;
    color: #111827;
    margin-bottom: 6px;
}

.section-subtitle {
    font-size: 15px;
    color: #64748b;
    margin-bottom: 18px;
}

.plan-box {
    background: #eef2ff;
    padding: 16px;
    border-radius: 18px;
    border-left: 5px solid #4f46e5;
    margin-bottom: 12px;
}

.memory-box {
    background: #f8fafc;
    padding: 16px;
    border-radius: 18px;
    border-left: 5px solid #2563eb;
    margin-bottom: 12px;
}

.source-chip {
    display: inline-block;
    background: #dcfce7;
    color: #166534;
    font-weight: 800;
    padding: 7px 12px;
    border-radius: 999px;
    margin-bottom: 10px;
}

.warning-chip {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    font-weight: 800;
    padding: 7px 12px;
    border-radius: 999px;
    margin-bottom: 10px;
}

.stButton>button {
    width: 100%;
    border-radius: 16px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-weight: 900;
    padding: 14px;
    border: none;
}

.stDownloadButton>button {
    width: 100%;
    border-radius: 16px;
    font-weight: 900;
    padding: 14px;
}

[data-testid="stFileUploader"] {
    background: white;
    padding: 14px;
    border-radius: 20px;
    border: 1px dashed #94a3b8;
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
    st.session_state.indexed_chunks = 0
    st.session_state.planner = {}
    st.session_state.workflow_status = {
        "Planner Agent": "Waiting",
        "Vector RAG Agent": "Waiting",
        "Research Agent": "Waiting",
        "Summary Agent": "Waiting",
        "Report Agent": "Waiting"
    }


def show_agent_card(agent_name, emoji):
    status = st.session_state.workflow_status[agent_name]

    st.markdown(
        f"""
        <div class="agent-card">
            <div class="agent-icon">{emoji}</div>
            <div class="agent-title">{agent_name}</div>
            <div class="agent-status">{status}</div>
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


st.markdown(
    """
    <div class="hero">
        <div class="badge">Advanced Agentic RAG SaaS Dashboard</div>
        <div class="title">Multi-Agent Research Assistant</div>
        <div class="subtitle">
            Upload documents, build a local vector database, retrieve source-grounded context,
            let an autonomous planner choose the workflow, and generate research reports using specialized AI agents.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


completed = count_completed_agents()
total_agents = 5
confidence = rag_confidence()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{total_agents}</div>
            <div class="metric-label">Agent Workflow</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{completed}</div>
            <div class="metric-label">Completed Steps</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.indexed_chunks}</div>
            <div class="metric-label">Vector DB Chunks</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{confidence}</div>
            <div class="metric-label">RAG Confidence</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.progress(completed / total_agents)

st.markdown("## Autonomous Agent Workflow")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    show_agent_card("Planner Agent", "🧭")

with c2:
    show_agent_card("Vector RAG Agent", "🧬")

with c3:
    show_agent_card("Research Agent", "🔎")

with c4:
    show_agent_card("Summary Agent", "📝")

with c5:
    show_agent_card("Report Agent", "📄")


st.markdown("---")

left, right = st.columns([1, 2])

with left:
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Research Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter a topic and upload a PDF/TXT document for vector database RAG.</div>',
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

    top_k = st.slider("Number of source chunks to retrieve", 2, 6, 4)

    run_button = st.button("Run Advanced Agentic RAG")
    clear_button = st.button("Clear Workspace")

    if clear_button:
        reset_outputs()
        st.session_state.document_text = ""
        st.success("Workspace cleared.")

    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Advanced Features Added</div>', unsafe_allow_html=True)

    st.markdown("""
This Day 6 version makes the project stronger:

- **Vector Database RAG:** stores document chunks inside ChromaDB.
- **Gemini Embeddings:** converts text into semantic vectors.
- **Autonomous Planner:** decides whether to use Vector RAG or topic-only GenAI.
- **Source Grounding:** agents answer using retrieved document chunks.
- **Agent Memory:** stores recent research topics, mode, trust level, and confidence.
- **SaaS UI:** professional dashboard layout.
""")

    if confidence >= 0.7:
        st.markdown('<div class="source-chip">High Source Confidence</div>', unsafe_allow_html=True)

    elif confidence > 0:
        st.markdown('<div class="warning-chip">Moderate / Low Source Confidence</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="warning-chip">No RAG Source Used Yet</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


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
            chunks = split_text_into_chunks(
                st.session_state.document_text,
                chunk_size=700
            )

            with st.spinner("Building ChromaDB vector database with Gemini embeddings..."):
                st.session_state.indexed_chunks = index_document_chunks(chunks)

            with st.spinner("Retrieving source chunks from vector database..."):
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
            st.session_state.summary = summary_agent(st.session_state.research)
            st.session_state.notes = notes_agent(st.session_state.research)
            st.session_state.questions = question_agent(st.session_state.research)

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

        st.success("Advanced Agentic RAG research completed successfully!")
        st.rerun()


st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧭 Planner",
    "🧬 Vector Sources",
    "🔎 Research",
    "📝 Summary",
    "📚 Notes",
    "❓ Questions",
    "📄 Report"
])

with tab1:
    st.markdown("### Autonomous Planner Output")

    if st.session_state.planner:
        st.markdown(
            f"""
            <div class="plan-box">
                <b>Mode:</b> {st.session_state.planner.get("mode")}<br>
                <b>Trust Level:</b> {st.session_state.planner.get("trust_level")}<br>
                <b>Reason:</b> {st.session_state.planner.get("reason")}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Planned Steps")

        for step in st.session_state.planner.get("steps", []):
            st.write(f"✅ {step}")

    else:
        st.info("Planner output will appear here.")


with tab2:
    st.markdown("### Retrieved Vector Database Sources")

    if st.session_state.retrieved_chunks:
        for item in st.session_state.retrieved_chunks:
            with st.expander(f"{item['source']} | Similarity Score: {item['score']}"):
                st.write(item["chunk"])

    elif st.session_state.retrieved_context:
        st.info(st.session_state.retrieved_context)

    else:
        st.info("Vector database sources will appear here.")


with tab3:
    st.markdown("### Research Agent Output")

    if st.session_state.research:
        st.markdown(st.session_state.research)
    else:
        st.info("Research output will appear here.")


with tab4:
    st.markdown("### Summary Agent Output")

    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.info("Summary output will appear here.")


with tab5:
    st.markdown("### Notes Agent Output")

    if st.session_state.notes:
        st.markdown(st.session_state.notes)
    else:
        st.info("Notes output will appear here.")


with tab6:
    st.markdown("### Question Agent Output")

    if st.session_state.questions:
        st.markdown(st.session_state.questions)
    else:
        st.info("Questions will appear here.")


with tab7:
    st.markdown("### Final Research Report")

    if st.session_state.report:
        st.markdown(st.session_state.report)

        st.download_button(
            label="Download Agentic RAG Report",
            data=st.session_state.report,
            file_name="agentic_rag_research_report.txt",
            mime="text/plain"
        )

    else:
        st.info("Final report will appear here.")


st.markdown("---")
st.markdown("## Agent Memory")

if st.session_state.memory:
    for item in st.session_state.memory:
        st.markdown(
            f"""
            <div class="memory-box">
                <b>Topic:</b> {item["topic"]}<br>
                <b>Research Mode:</b> {item["mode"]}<br>
                <b>Trust Level:</b> {item["trust"]}<br>
                <b>Sources Used:</b> {item["sources"]}<br>
                <b>RAG Confidence:</b> {item["confidence"]}
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("Agent memory will appear after running research.")


st.markdown("---")
st.markdown("## Day 6 Completed Features")

st.success("""
Advanced vector database RAG, autonomous planner agent, source-grounded research,
agent memory, confidence scoring, and SaaS-style UI added successfully.
""")
