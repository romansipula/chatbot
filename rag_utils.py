import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import tiktoken
import pandas as pd
import fitz  # PyMuPDF
import docx2txt

# Utility functions for RAG

def load_txt(file):
    return file.read().decode('utf-8')

def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_docx(file):
    with open("temp.docx", "wb") as f:
        f.write(file.read())
    text = docx2txt.process("temp.docx")
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def embed_chunks(chunks, model):
    return model.encode(chunks, show_progress_bar=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(query, model, index, chunks, top_k=3):
    query_emb = model.encode([query])
    D, I = index.search(query_emb, top_k)
    return [chunks[i] for i in I[0]]
