def extract_text_from_pdf(uploaded_file):
    try:
        import pdfplumber
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pdfplumber is required for PDF uploads. Install it with: pip install pdfplumber"
        ) from exc

    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")


def split_text_into_chunks(text, chunk_size=900):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
