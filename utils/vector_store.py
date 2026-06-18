import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "multi_agent_research_docs"


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


def get_embedding(text):
    """
    Converts text into a numerical embedding vector using Gemini.
    This vector is used for semantic search in ChromaDB.
    """

    client = get_gemini_client()

    if not client:
        return None

    clean_text = text[:8000]

    models = [
        "gemini-embedding-001"
    ]

    for model in models:
        try:
            result = client.models.embed_content(
                model=model,
                contents=clean_text
            )

            embedding = result.embeddings[0]

            if hasattr(embedding, "values"):
                return embedding.values

            return embedding

        except Exception:
            continue

    return None


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def reset_vector_database():
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def index_document_chunks(chunks):
    """
    Stores document chunks in ChromaDB with Gemini embeddings.
    """

    collection = reset_vector_database()

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
            "source": f"Source Chunk {index}"
        })

    if documents:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    return len(documents)


def query_vector_database(query, top_k=4):
    """
    Searches ChromaDB using Gemini query embedding.
    """

    client = get_chroma_client()

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return []

    query_embedding = get_embedding(query)

    if query_embedding is None:
        return []

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
            "source": meta.get("source", "Source Chunk"),
            "chunk_id": meta.get("chunk_id", 0)
        })

    return retrieved_chunks


def combine_vector_chunks(retrieved_chunks):
    context = ""

    for item in retrieved_chunks:
        context += f"\n\n[{item['source']} | Similarity Score: {item['score']}]\n"
        context += item["chunk"]

    return context
