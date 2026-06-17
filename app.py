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

from utils.retriever import (
    retrieve_relevant_chunks,
    combine_retrieved_chunks
)

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
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
        radial-gradient(circle at top left, #dbeafe 0, transparent 35%),
        radial-gradient(circle at top right, #ede9fe 0, transparent 30%),
        linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero-card {
    background: linear-gradient(135deg, #111827, #1e3a8a, #4c1d95);
    padding: 36px;
    border-radius: 28px;
    color: white;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.25);
    margin-bottom: 28px;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 14px;
}

.main-title {
    font-size: 48px;
    font-weight: 900;
    margin-bottom: 10px;
}

.sub-title {
    font-size: 18px;
    color: #e5e7eb;
    max-width: 900px;
}

.glass-card {
    background: rgba(255,255,255,0.88);
    backdrop-filter: blur(14px);
    padding: 24px;
    border-radius: 24px;
    border: 1px solid rgba(226,232,240,0.9);
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    margin-bottom: 22px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 22px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
}

.metric-number {
    font-size: 34px;
    font-weight: 900;
    color: #2563eb;
}

.metric-label {
    font-size: 14px;
    color: #64748b;
    font-weight: 600;
}

.agent-card {
    background: linear-gradient(180deg, #ffffff, #f8fafc);
    padding: 20px;
    border-radius: 22px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
    text-align: center;
    min-height: 150px;
}

.agent-icon {
    font-size: 30px;
    margin-bottom: 8px;
}

.agent-title {
    font-size: 15px;
    font-weight: 800;
    color: #111827;
}

.agent-status {
    margin-top: 10px;
    font-size: 13px;
    font-weight: 800;
    color: #2563eb;
    background: #dbeafe;
    padding: 6px 10px;
    border-radius: 999px;
    display: inline-block;
}

.section-title {
    font-size: 24px;
    font-weight: 900;
    color: #111827;
    margin-bottom: 8px;
}

.section-subtitle {
    font-size: 15px;
    color: #64748b;
    margin-bottom: 18px;
}

.memory-box {
    background: #f8fafc;
    padding: 16px;
    border-radius: 16px;
    border-left: 5px solid #2563eb;
    margin-bottom: 10px;
}

.source-box {
    background: #ffffff;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    margin-bottom: 12px;
}

.stButton>button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-weight: 800;
    padding: 13px;
    border: none;
}

.stDownloadButton>button {
    width: 100%;
    border-radius: 14px;
    font-weight: 800;
    padding: 13px;
}

[data-testid="stFileUploader"] {
    background: white;
    padding: 14px;
    border-radius: 18px;
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
        "memory": [],
        "history": [],
        "workflow_status": {
            "Research Agent": "Waiting",
            "Summary Agent": "Waiting",
            "Notes Agent": "Waiting",
            "Question Agent": "Waiting",
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
    st.session_state.workflow_status = {
        "Research Agent": "Waiting",
        "Summary Agent": "Waiting",
        "Notes Agent": "Waiting",
        "Question Agent": "Waiting",
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
        1 for status in st.session_state.workflow_status.values()
        if status == "Completed"
    )


def average_similarity_score():
    if not st.session_state.retrieved_chunks:
        return 0

    total_score = sum(item["score"] for item in st.session_state.retrieved_chunks)
    return round(total_score / len(st.session_state.retrieved_chunks), 3)


def add_memory(topic):
    memory_item = {
        "topic": topic,
        "sources_used": len(st.session_state.retrieved_chunks),
        "confidence": average_similarity_score()
    }

    st.session_state.memory.insert(0, memory_item)
    st.session_state.memory = st.session_state.memory[:5]


init_session_state()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">Agentic AI + RAG Research Workspace</div>
        <div class="main-title">Multi-Agent Research Assistant</div>
        <div class="sub-title">
            A professional Generative AI research dashboard where multiple AI agents collaborate with
            RAG-based document retrieval to create research summaries, notes, questions, and final reports.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

completed_agents = count_completed_agents()
total_agents = 5
progress_value = completed_agents / total_agents
avg_score = average_similarity_score()

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{total_agents}</div>
            <div class="metric-label">Specialized Agents</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{completed_agents}</div>
            <div class="metric-label">Completed Agents</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{len(st.session_state.retrieved_chunks)}</div>
            <div class="metric-label">Retrieved Sources</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{avg_score}</div>
            <div class="metric-label">RAG Confidence</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.progress(progress_value)

st.markdown("## Agent Workflow")

a1, a2, a3, a4, a5 = st.columns(5)

with a1:
    show_agent_card("Research Agent", "🔎")

with a2:
    show_agent_card("Summary Agent", "📝")

with a3:
    show_agent_card("Notes Agent", "📚")

with a4:
    show_agent_card("Question Agent", "❓")

with a5:
    show_agent_card("Report Agent", "📄")

st.markdown("---")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Research Control Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter a topic and optionally upload a PDF/TXT document for RAG-based research.</div>',
        unsafe_allow_html=True
    )

    topic = st.text_input(
        "Research Topic",
        placeholder="Example: Retrieval Augmented Generation"
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT for RAG",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            st.session_state.document_text = extract_text_from_pdf(uploaded_file)
        else:
            st.session_state.document_text = extract_text_from_txt(uploaded_file)

        st.success("Document processed successfully.")
        st.caption(f"Extracted words: {len(st.session_state.document_text.split())}")

    run_button = st.button("Run Agentic Research")
    clear_button = st.button("Clear Workspace")

    if clear_button:
        reset_outputs()
        st.success("Workspace cleared.")

    st.markdown("### Suggested Topics")
    st.info("Retrieval Augmented Generation")
    st.info("AI Agents")
    st.info("Vector Databases")
    st.info("Transformer Architecture")

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How This Advanced Workflow Works</div>', unsafe_allow_html=True)

    st.markdown("""
This version follows a stronger **Agentic + RAG workflow**:

1. The user enters a topic.
2. The document is split into smaller chunks.
3. The retriever selects the most relevant chunks using TF-IDF similarity.
4. The Research Agent uses the retrieved context.
5. Other agents transform the research into summary, notes, questions, and a final report.
6. The app stores recent research activity as simple agent memory.
""")

    st.success("Day 5 Upgrade: Professional UI, RAG confidence, source tracking, and agent memory added.")

    st.markdown('</div>', unsafe_allow_html=True)

if run_button:
    if not topic.strip():
        st.error("Please enter a research topic.")
    else:
        reset_outputs()

        if topic not in st.session_state.history:
            st.session_state.history.append(topic)

        context = ""

        if st.session_state.document_text:
            chunks = split_text_into_chunks(st.session_state.document_text)
            retrieved_chunks = retrieve_relevant_chunks(topic, chunks, top_k=3)
            context = combine_retrieved_chunks(retrieved_chunks)

            st.session_state.retrieved_chunks = retrieved_chunks
            st.session_state.retrieved_context = context
        else:
            st.session_state.retrieved_context = "No document uploaded. Agents used topic-based GenAI only."

        st.session_state.workflow_status["Research Agent"] = "Running"
        with st.spinner("Research Agent is analyzing topic and sources..."):
            st.session_state.research = research_agent(topic, context)
        st.session_state.workflow_status["Research Agent"] = "Completed"

        st.session_state.workflow_status["Summary Agent"] = "Running"
        with st.spinner("Summary Agent is creating short summary..."):
            st.session_state.summary = summary_agent(st.session_state.research)
        st.session_state.workflow_status["Summary Agent"] = "Completed"

        st.session_state.workflow_status["Notes Agent"] = "Running"
        with st.spinner("Notes Agent is preparing study notes..."):
            st.session_state.notes = notes_agent(st.session_state.research)
        st.session_state.workflow_status["Notes Agent"] = "Completed"

        st.session_state.workflow_status["Question Agent"] = "Running"
        with st.spinner("Question Agent is generating questions..."):
            st.session_state.questions = question_agent(st.session_state.research)
        st.session_state.workflow_status["Question Agent"] = "Completed"

        st.session_state.workflow_status["Report Agent"] = "Running"
        with st.spinner("Report Agent is creating final report..."):
            st.session_state.report = report_agent(
                topic,
                st.session_state.research,
                st.session_state.summary,
                st.session_state.notes,
                st.session_state.questions,
                context
            )
        st.session_state.workflow_status["Report Agent"] = "Completed"

        add_memory(topic)

        st.success("Agentic research completed successfully!")
        st.rerun()

st.markdown("---")

st.markdown("## Agent Outputs")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔎 Research",
    "📝 Summary",
    "📚 Notes",
    "❓ Questions",
    "📄 Final Report",
    "📌 RAG Sources",
    "🧠 Agent Memory"
])

with tab1:
    st.markdown("### Research Agent Output")
    if st.session_state.research:
        st.markdown(st.session_state.research)
    else:
        st.info("Research output will appear here.")

with tab2:
    st.markdown("### Summary Agent Output")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.info("Summary output will appear here.")

with tab3:
    st.markdown("### Notes Agent Output")
    if st.session_state.notes:
        st.markdown(st.session_state.notes)
    else:
        st.info("Notes output will appear here.")

with tab4:
    st.markdown("### Question Agent Output")
    if st.session_state.questions:
        st.markdown(st.session_state.questions)
    else:
        st.info("Questions will appear here.")

with tab5:
    st.markdown("### Final Research Report")
    if st.session_state.report:
        st.markdown(st.session_state.report)

        st.download_button(
            label="Download Final Report",
            data=st.session_state.report,
            file_name="multi_agent_research_report.txt",
            mime="text/plain"
        )
    else:
        st.info("Final report will appear here.")

with tab6:
    st.markdown("### Retrieved RAG Sources")

    if st.session_state.retrieved_chunks:
        for index, item in enumerate(st.session_state.retrieved_chunks, start=1):
            with st.expander(f"Source Chunk {index} | Similarity Score: {item['score']}"):
                st.write(item["chunk"])
    elif st.session_state.retrieved_context:
        st.info(st.session_state.retrieved_context)
    else:
        st.info("Retrieved source chunks will appear here.")

with tab7:
    st.markdown("### Agent Memory")

    if st.session_state.memory:
        for item in st.session_state.memory:
            st.markdown(
                f"""
                <div class="memory-box">
                    <b>Topic:</b> {item["topic"]}<br>
                    <b>Sources Used:</b> {item["sources_used"]}<br>
                    <b>RAG Confidence:</b> {item["confidence"]}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Agent memory will appear after running research.")

st.markdown("---")

st.markdown("## Day 5 Completed Features")

st.write("""
- Added professional AI dashboard UI
- Added advanced color theme and glass cards
- Added RAG confidence metric
- Added retrieved source tracking
- Added agent memory panel
- Improved agent workflow design
- Improved project quality for GitHub and LinkedIn
""")