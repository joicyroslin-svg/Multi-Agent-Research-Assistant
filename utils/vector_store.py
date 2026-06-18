import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "multi_agent_research_docs"
FALLBACK_CHUNKS = []


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


def get_gemini_client():
    api_key = get_api_key()

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


def get_chromadb():
    try:
        import chromadb

        return chromadb
    except Exception:
        return None


def get_embedding(text):
    """
    Converts text into an embedding vector using Gemini.
    Returns None when embeddings are unavailable, so the app can use TF-IDF fallback.
    """

    client = get_gemini_client()

    if not client:
        return None

    clean_text = text[:8000]

    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=clean_text
        )

        embedding = result.embeddings[0]

        if hasattr(embedding, "values"):
            return embedding.values

        return embedding

    except Exception:
        return None


def get_chroma_client():
    chromadb = get_chromadb()

    if chromadb is None:
        return None

    return chromadb.PersistentClient(path=CHROMA_PATH)


def reset_vector_database():
    client = get_chroma_client()

    if client is None:
        return None

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def index_document_chunks(chunks):
    """
    Stores chunks in ChromaDB when available.
    On Streamlit Cloud/Python versions where ChromaDB fails to import, stores chunks
    in memory and uses TF-IDF retrieval instead of crashing the app.
    """

    global FALLBACK_CHUNKS
    FALLBACK_CHUNKS = chunks

    collection = reset_vector_database()

    if collection is None:
        return len(chunks)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    def embed_chunk(index, chunk):
        return index, chunk, get_embedding(chunk)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(embed_chunk, index, chunk)
            for index, chunk in enumerate(chunks, start=1)
        ]

        embedded_chunks = []

        for future in as_completed(futures):
            embedded_chunks.append(future.result())

    embedded_chunks.sort(key=lambda item: item[0])

    for index, chunk, embedding in embedded_chunks:
        if embedding is None:
            continue

        ids.append(str(uuid.uuid4()))
        documents.append(chunk)
        embeddings.append(embedding)
        metadatas.append({
            "chunk_id": index,
            "source": f"Evidence Section {index}"
        })

    if not documents:
        return len(chunks)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(documents)


def query_fallback_database(query, top_k=4):
    if not query or not FALLBACK_CHUNKS:
        return []

    documents = [query] + FALLBACK_CHUNKS
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    query_vector = tfidf_matrix[0:1]
    chunk_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(query_vector, chunk_vectors).flatten()

    scored_chunks = []

    for index, score in enumerate(scores, start=1):
        scored_chunks.append({
            "chunk": FALLBACK_CHUNKS[index - 1],
            "score": round(float(score), 3),
            "source": f"Evidence Section {index}",
            "chunk_id": index
        })

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]


def query_vector_database(query, top_k=4):
    """
    Searches ChromaDB when available, otherwise uses TF-IDF fallback retrieval.
    """

    client = get_chroma_client()

    if client is None:
        return query_fallback_database(query, top_k)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return query_fallback_database(query, top_k)

    query_embedding = get_embedding(query)

    if query_embedding is None:
        return query_fallback_database(query, top_k)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        similarity_score = round(1 - float(distance), 3)

        retrieved_chunks.append({
            "chunk": doc,
            "score": similarity_score,
            "source": meta.get("source", "Evidence Section"),
            "chunk_id": meta.get("chunk_id", 0)
        })

    return retrieved_chunks


def combine_vector_chunks(retrieved_chunks):
    context = ""

    for item in retrieved_chunks:
        context += f"\n\n[{item['source']} | Similarity Score: {item['score']}]\n"
        context += item["chunk"]

    return context
