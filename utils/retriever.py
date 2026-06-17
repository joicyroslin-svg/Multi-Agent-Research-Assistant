def retrieve_relevant_chunks(topic, chunks, top_k=3):
    topic_words = set(topic.lower().split())
    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(topic_words.intersection(chunk_words))
        scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    top_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    if not top_chunks:
        return chunks[:top_k]

    return top_chunks