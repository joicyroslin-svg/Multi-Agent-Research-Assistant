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

    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

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


def research_agent(topic, context=""):
    prompt = f"""
You are a Research Agent.

Topic:
{topic}

Retrieved Document Context:
{context}

Task:
Explain the topic using the retrieved document context.
If the context is useful, base your answer on it.
If the context is limited, explain using general knowledge and mention that document context was limited.

Format:
## Topic Overview
## Key Concepts
## Real-World Applications
## Why It Is Important

Keep it beginner-friendly.
"""
    return ask_gemini(prompt)


def summary_agent(research_output):
    prompt = f"""
You are a Summary Agent.

Convert this research content into a short summary:

{research_output}

Format:
## Short Summary
Give 5 simple bullet points.

## One-Line Meaning
Give one easy line.
"""
    return ask_gemini(prompt)


def notes_agent(research_output):
    prompt = f"""
You are a Notes Agent.

Convert this research content into easy notes:

{research_output}

Format:
## Easy Notes
## Important Terms
## Easy Revision Line
"""
    return ask_gemini(prompt)


def question_agent(research_output):
    prompt = f"""
You are a Question Generator Agent.

Create questions from this content:

{research_output}

Format:
## 2-Mark Questions
Give 5 questions.

## 5-Mark Questions
Give 5 questions.

## Interview Questions
Give 5 questions.
"""
    return ask_gemini(prompt)


def report_agent(topic, research_output, summary_output, notes_output, questions_output, context=""):
    prompt = f"""
You are a Report Agent.

Create a final student-friendly research report.

Topic:
{topic}

Retrieved Document Context:
{context}

Research:
{research_output}

Summary:
{summary_output}

Notes:
{notes_output}

Questions:
{questions_output}

Format:
# Research Report

## Topic
## Introduction
## Explanation
## Summary
## Study Notes
## Practice Questions
## Conclusion

Mention if uploaded document context was used.
"""
    return ask_gemini(prompt)