import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]

    except Exception:
        pass

    return None


def ask_gemini(prompt):
    api_key = get_api_key()

    if not api_key:
        return "Gemini API key is missing. Add GEMINI_API_KEY in your .env file."

    client = genai.Client(api_key=api_key)

    models = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-1.5-flash"
    ]

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            if response.text:
                return response.text

        except Exception:
            continue

    return "AI response failed. Please check your API key or internet connection."


def research_agent(topic, context="", plan=None):
    plan_text = plan if plan else {}

    prompt = f"""
You are a Research Agent in an Agentic RAG system.

Topic:
{topic}

Planner Decision:
{plan_text}

Retrieved Vector Database Context:
{context}

Your task:
Create a source-grounded research explanation.

Rules:
- Use the retrieved context when it is available.
- Mention when the answer is based on uploaded document context.
- If context is weak or missing, clearly say the answer is based on general knowledge.
- Do not invent fake sources.
- Keep language simple for students.

Format:
## Topic Overview
## Source-Grounded Explanation
## Key Concepts
## Real-World Applications
## Why It Is Important
## Source Usage Note
"""

    return ask_gemini(prompt)


def summary_agent(research_output):
    prompt = f"""
You are a Summary Agent.

Summarize this research output in simple language:

{research_output}

Format:
## Short Summary
Give 5-7 simple bullet points.

## One-Line Meaning
Give one easy one-line meaning.

## Important Takeaway
Give the most important point.
"""

    return ask_gemini(prompt)


def notes_agent(research_output):
    prompt = f"""
You are a Notes Agent.

Convert this research into exam/interview-friendly notes:

{research_output}

Format:
## Easy Notes
## Important Terms
## Easy Revision Line
## Interview Revision Points
"""

    return ask_gemini(prompt)


def question_agent(research_output):
    prompt = f"""
You are a Question Generator Agent.

Create questions from this research:

{research_output}

Format:
## 2-Mark Questions
Give 5 questions.

## 5-Mark Questions
Give 5 questions.

## Interview Questions
Give 5 questions.

## Practical Project Questions
Give 3 project-based questions.
"""

    return ask_gemini(prompt)


def report_agent(
    topic,
    research_output,
    summary_output,
    notes_output,
    questions_output,
    context="",
    plan=None
):
    plan_text = plan if plan else {}

    prompt = f"""
You are a Final Report Agent.

Create a professional research report for a student portfolio.

Topic:
{topic}

Planner Decision:
{plan_text}

Retrieved Vector Database Context:
{context}

Research:
{research_output}

Summary:
{summary_output}

Notes:
{notes_output}

Questions:
{questions_output}

Rules:
- Mention whether vector database RAG was used.
- Mention if the report is grounded in uploaded document context.
- Do not create fake citations.
- Keep it clear, structured, and useful.

Format:
# Agentic RAG Research Report
## Topic
## Research Mode Used
## Introduction
## Source-Grounded Explanation
## Summary
## Study Notes
## Practice Questions
## Source Usage
## Conclusion
"""

    return ask_gemini(prompt)
