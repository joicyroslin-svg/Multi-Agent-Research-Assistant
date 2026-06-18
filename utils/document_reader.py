import pdfplumber


def extract_text_from_pdf(uploaded_file):
    text = ""

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        return text.strip()

    except Exception as e:
        return f"PDF reading error: {e}"


def extract_text_from_txt(uploaded_file):
    try:
        bytes_data = uploaded_file.read()
        text = bytes_data.decode("utf-8", errors="ignore")
        return text.strip()

    except Exception as e:
        return f"TXT reading error: {e}"


def split_text_into_chunks(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []

    if not words:
        return chunks

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

        if start < 0:
            start = 0

        if start >= len(words):
            break

    return chunks
