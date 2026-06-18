def planner_agent(topic, has_document, retrieved_chunks):
    """
    Simple autonomous planner.
    It decides whether to use Vector RAG or topic-only GenAI.
    """

    plan = {
        "mode": "",
        "steps": [],
        "trust_level": "",
        "reason": ""
    }

    if has_document and retrieved_chunks:
        avg_score = sum(item["score"] for item in retrieved_chunks) / len(retrieved_chunks)

        plan["mode"] = "Vector RAG Research Mode"
        plan["steps"] = [
            "Use vector database retrieval",
            "Ground research using retrieved document chunks",
            "Generate summary from source-grounded research",
            "Create notes and questions",
            "Prepare final report with source awareness"
        ]

        if avg_score >= 0.70:
            plan["trust_level"] = "High"
            plan["reason"] = "Relevant document chunks were found with strong similarity."

        elif avg_score >= 0.40:
            plan["trust_level"] = "Medium"
            plan["reason"] = "Some useful chunks were found, but the document match is moderate."

        else:
            plan["trust_level"] = "Low"
            plan["reason"] = "Retrieved chunks have weak similarity, so output should be checked carefully."

    elif has_document and not retrieved_chunks:
        plan["mode"] = "Fallback Research Mode"
        plan["steps"] = [
            "Document was uploaded",
            "No strong matching chunks were found",
            "Use Gemini general knowledge",
            "Clearly mention limited document grounding"
        ]
        plan["trust_level"] = "Low"
        plan["reason"] = "Document exists, but retrieval did not find useful context."

    else:
        plan["mode"] = "Topic-Only GenAI Mode"
        plan["steps"] = [
            "No document uploaded",
            "Use Gemini topic knowledge",
            "Generate research, notes, questions, and report",
            "Mention that no uploaded document source was used"
        ]
        plan["trust_level"] = "General"
        plan["reason"] = "No uploaded document was available for RAG grounding."

    return plan
