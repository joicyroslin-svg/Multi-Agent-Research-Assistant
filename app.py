import streamlit as st

from utils.agents import (
    research_agent,
    summary_agent,
    notes_agent,
    question_agent,
    report_agent
)

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
}

.sub-title {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 25px;
}

.agent-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 18px;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    background-color: #2563eb;
    color: white;
    font-weight: 700;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Multi-Agent Research Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A GenAI app where multiple AI agents research, summarize, create notes, generate questions, and prepare a final report.</div>',
    unsafe_allow_html=True
)

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

st.markdown("## Enter Research Topic")

topic = st.text_input(
    "Topic",
    placeholder="Example: Retrieval Augmented Generation"
)

run_button = st.button("Run Multi-Agent Research")

if run_button:
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        with st.spinner("Research Agent is working..."):
            st.session_state.research = research_agent(topic)

        with st.spinner("Summary Agent is working..."):
            st.session_state.summary = summary_agent(st.session_state.research)

        with st.spinner("Notes Agent is working..."):
            st.session_state.notes = notes_agent(st.session_state.research)

        with st.spinner("Question Agent is working..."):
            st.session_state.questions = question_agent(st.session_state.research)

        with st.spinner("Report Agent is preparing final report..."):
            st.session_state.report = report_agent(
                topic,
                st.session_state.research,
                st.session_state.summary,
                st.session_state.notes,
                st.session_state.questions
            )

        st.success("Research completed successfully!")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Research Agent",
    "Summary Agent",
    "Notes Agent",
    "Question Agent",
    "Final Report"
])

with tab1:
    st.markdown("## Research Agent Output")
    if st.session_state.research:
        st.markdown(st.session_state.research)
    else:
        st.info("Research output will appear here.")

with tab2:
    st.markdown("## Summary Agent Output")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.info("Summary output will appear here.")

with tab3:
    st.markdown("## Notes Agent Output")
    if st.session_state.notes:
        st.markdown(st.session_state.notes)
    else:
        st.info("Notes output will appear here.")

with tab4:
    st.markdown("## Question Agent Output")
    if st.session_state.questions:
        st.markdown(st.session_state.questions)
    else:
        st.info("Questions will appear here.")

with tab5:
    st.markdown("## Final Report")
    if st.session_state.report:
        st.markdown(st.session_state.report)

        st.download_button(
            label="Download Report",
            data=st.session_state.report,
            file_name="research_report.txt",
            mime="text/plain"
        )
    else:
        st.info("Final report will appear here.")

st.markdown("---")

st.markdown("## Day 1 Completed")
st.write("""
- Created Streamlit app
- Connected Gemini API
- Added Research Agent
- Added Summary Agent
- Added Notes Agent
- Added Question Agent
- Added Report Agent
- Added final report download
""")
