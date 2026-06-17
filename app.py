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

from utils.retriever import retrieve_relevant_chunks, combine_retrieved_chunks

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 6px;
}

.sub-title {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 28px;
}

.hero-card {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
    padding: 30px;
    border-radius: 24px;
    border: 1px solid #e5e7eb;
    margin-bottom: 24px;
}

.section-card {
    background: #ffffff;
    padding: 22px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.agent-card {
    background: #f9fafb;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    text-align: center;
}

.agent-title {
    font-size: 16px;
    font-weight: 700;
    color: #111827;
}

.agent-status {
    font-size: 14px;
    color: #2563eb;
    font-weight: 600;
    margin-top: 8px;
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
    color: #2563eb;
}

.metric-label {
    font-size: 14px;
    color: #6b7280;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    background-color: #2563eb;
    color: white;
    font-weight: 700;
    padding: 12px;
    border: none;
}

.stDownloadButton>button {
    width: 100%;
    border-radius: 12px;
    font-weight: 700;
    padding: 12px;
}

.output-box {
    background: #ffffff;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "research" not in st.session_state:
        st.session_state.research = ""

    if "summary" not in st.session_state:
        st.session_state.summary = ""

    if "notes" not in st.session_state:
        st.session_state.notes = ""

    if "questions" not in st.session_state:
        st.session_state.questions = ""

    if "report" not in st.session_state:
        st.session_state.report = ""

    if "workflow_status" not in st.session_state:
        st.session_state.workflow_status = {
            "Research Agent": "Waiting",
            "Summary Agent": "Waiting",
            "Notes Agent": "Waiting",
            "Question Agent": "Waiting",
            "Report Agent": "Waiting"
        }

    if "history" not in st.session_state:
        st.session_state.history = []
    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    if "retrieved_context" not in st.session_state:
        st.session_state.retrieved_context = ""
    if "retrieved_chunks" not in st.session_state:
        st.session_state.retrieved_chunks = []        


def reset_outputs():
    st.session_state.research = ""
    st.session_state.summary = ""
    st.session_state.notes = ""
    st.session_state.questions = ""
    st.session_state.report = ""
    st.session_state.retrieved_chunks = []
    st.session_state.retrieved_context = ""
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
            <div style="font-size: 28px;">{emoji}</div>
            <div class="agent-title">{agent_name}</div>
            <div class="agent-status">{status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def count_completed_agents():
    completed = 0

    for status in st.session_state.workflow_status.values():
        if status == "Completed":
            completed += 1

    return completed


init_session_state()

st.markdown(
    """
    <div class="hero-card">
        <div class="main-title">Multi-Agent Research Assistant</div>
        <div class="sub-title">
            A Generative AI app where multiple AI agents work together to research a topic, summarize it,
            create notes, generate questions, and prepare a final report.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Top metrics
completed_agents = count_completed_agents()
total_agents = 5
progress_value = completed_agents / total_agents

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="metric-number">{total_agents}</div>
            <div class="metric-label">AI Agents</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="metric-number">{completed_agents}</div>
            <div class="metric-label">Completed Agents</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="metric-number">{len(st.session_state.history)}</div>
            <div class="metric-label">Research Topics</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.progress(progress_value)

st.markdown("## Agent Workflow")

agent_col1, agent_col2, agent_col3, agent_col4, agent_col5 = st.columns(5)

with agent_col1:
    show_agent_card("Research Agent", "🔎")

with agent_col2:
    show_agent_card("Summary Agent", "📝")

with agent_col3:
    show_agent_card("Notes Agent", "📚")

with agent_col4:
    show_agent_card("Question Agent", "❓")

with agent_col5:
    show_agent_card("Report Agent", "📄")

st.markdown("---")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("## Research Control Panel")

    topic = st.text_input(
        "Enter research topic",
        placeholder="Example: Retrieval Augmented Generation"
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT document for RAG",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            st.session_state.document_text = extract_text_from_pdf(uploaded_file)
        else:
            st.session_state.document_text = extract_text_from_txt(uploaded_file)

        st.success("Document uploaded and text extracted successfully.")
        st.write(f"Extracted words: {len(st.session_state.document_text.split())}")

    run_button = st.button("Run Multi-Agent Research")
    clear_button = st.button("Clear Outputs")

    if clear_button:
        reset_outputs()
        st.success("Outputs cleared successfully.")

    st.markdown("### Try these topics")
    st.info("Retrieval Augmented Generation")
    st.info("AI Agents")
    st.info("Transformer Architecture")
    st.info("Vector Databases")

with right_col:
    st.markdown("## Project Explanation")

    st.markdown(
        """
        This app uses different AI agents for different tasks.

        **Research Agent** explains the topic.  
        **Summary Agent** makes it short.  
        **Notes Agent** creates easy notes.  
        **Question Agent** generates questions.  
        **Report Agent** combines everything into a final report.
        """
    )

    st.markdown("### Day 2 Upgrade")
    st.success("Today we added a cleaner dashboard, agent workflow tracker, progress bar, and research history.")

if run_button:
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        reset_outputs()

        context = ""

        if st.session_state.document_text:
            chunks = split_text_into_chunks(st.session_state.document_text)
            retrieved_chunks = retrieve_relevant_chunks(topic, chunks, top_k=3)
            context = combine_retrieved_chunks(retrieved_chunks)

            st.session_state.retrieved_chunks = retrieved_chunks
            st.session_state.retrieved_context = context
        else:
            st.session_state.retrieved_chunks = []
            st.session_state.retrieved_context = "No document uploaded. Agents used topic-based GenAI only."

        if topic not in st.session_state.history:
            st.session_state.history.append(topic)

        st.session_state.workflow_status["Research Agent"] = "Running"
        with st.spinner("Research Agent is working..."):
            st.session_state.research = research_agent(topic, context)
        st.session_state.workflow_status["Research Agent"] = "Completed"

        st.session_state.workflow_status["Summary Agent"] = "Running"
        with st.spinner("Summary Agent is working..."):
            st.session_state.summary = summary_agent(st.session_state.research)
        st.session_state.workflow_status["Summary Agent"] = "Completed"

        st.session_state.workflow_status["Notes Agent"] = "Running"
        with st.spinner("Notes Agent is working..."):
            st.session_state.notes = notes_agent(st.session_state.research)
        st.session_state.workflow_status["Notes Agent"] = "Completed"

        st.session_state.workflow_status["Question Agent"] = "Running"
        with st.spinner("Question Agent is working..."):
            st.session_state.questions = question_agent(st.session_state.research)
        st.session_state.workflow_status["Question Agent"] = "Completed"

        st.session_state.workflow_status["Report Agent"] = "Running"
        with st.spinner("Report Agent is preparing final report..."):
            st.session_state.report = report_agent(
                topic,
                st.session_state.research,
                st.session_state.summary,
                st.session_state.notes,
                st.session_state.questions,
                context
            )
        st.session_state.workflow_status["Report Agent"] = "Completed"

        st.success("Multi-agent research completed successfully!")
        st.rerun()

st.markdown("---")

st.markdown("## Agent Outputs")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔎 Research",
    "📝 Summary",
    "📚 Notes",
    "❓ Questions",
    "📄 Final Report",
    "📌 Retrieved Context"
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
    st.markdown("### Final Report")
    if st.session_state.report:
        st.markdown(st.session_state.report)

        st.download_button(
            label="Download Final Research Report",
            data=st.session_state.report,
            file_name="multi_agent_research_report.txt",
            mime="text/plain"
        )
    else:
        st.info("Final report will appear here.")

with tab6:
    st.markdown("### Retrieved RAG Context")

    if st.session_state.retrieved_chunks:
        for index, item in enumerate(st.session_state.retrieved_chunks, start=1):
            with st.expander(f"Source Chunk {index} | Similarity Score: {item['score']}"):
                st.write(item["chunk"])
    elif st.session_state.retrieved_context:
        st.info(st.session_state.retrieved_context)
    else:
        st.info("Retrieved document context will appear here.")

st.markdown("---")

st.markdown("## Research History")

if st.session_state.history:
    for index, item in enumerate(st.session_state.history, start=1):
        st.write(f"{index}. {item}")
else:
    st.info("No research topics yet.")
