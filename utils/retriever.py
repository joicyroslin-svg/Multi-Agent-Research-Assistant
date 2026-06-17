from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_relevant_chunks(topic, chunks, top_k=3):
    """
    Retrieves the most relevant document chunks using TF-IDF + cosine similarity.
    """

    if not topic or not chunks:
        return []

    documents = [topic] + chunks

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    topic_vector = tfidf_matrix[0:1]
    chunk_vectors = tfidf_matrix[1:]

    similarity_scores = cosine_similarity(topic_vector, chunk_vectors).flatten()

    scored_chunks = []

    for index, score in enumerate(similarity_scores):
        scored_chunks.append({
            "chunk": chunks[index],
            "score": round(float(score), 3)
        })

    scored_chunks = sorted(
        scored_chunks,
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_chunks[:top_k]


def combine_retrieved_chunks(retrieved_chunks):
    """
    Combines retrieved chunks into one context text for Gemini.
    """

    context = ""

    for index, item in enumerate(retrieved_chunks, start=1):
        context += f"\n\nSource Chunk {index} | Similarity Score: {item['score']}\n"
        context += item["chunk"]

    return context