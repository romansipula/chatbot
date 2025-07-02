import streamlit as st
from openai import OpenAI
import os
import tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken
import pandas as pd
import fitz  # PyMuPDF
import docx2txt
from rag_utils import load_txt, load_pdf, load_docx, chunk_text, embed_chunks, build_faiss_index, search_index

# Professional HR Chatbot UI
st.set_page_config(page_title="HR Support Chatbot", page_icon="💼", layout="centered")
st.markdown("""
# 💼 HR Support Chatbot
Welcome to your HR assistant. Ask any questions about employee benefits, policies, leave, payroll, or other HR topics. Upload your HR handbook or policy documents in the sidebar to get answers based on your company's own information.
""")

# Try to get OpenAI API key from Streamlit secrets first
openai_api_key = st.secrets.get("openai_api_key", "")
if not openai_api_key:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # File uploader for knowledge source
    st.sidebar.header("Knowledge Source (RAG)")
    rag_file = st.sidebar.file_uploader("Upload a TXT, PDF, or DOCX file for RAG", type=["txt", "pdf", "docx"])

    if rag_file:
        # Load file content
        if rag_file.type == "text/plain":
            file_text = load_txt(rag_file)
        elif rag_file.type == "application/pdf":
            file_text = load_pdf(rag_file)
        elif rag_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            file_text = load_docx(rag_file)
        else:
            st.error("Unsupported file type.")
            file_text = None

        if file_text:
            # Chunk and embed
            chunks = chunk_text(file_text)
            if "embed_model" not in st.session_state:
                st.session_state["embed_model"] = SentenceTransformer("all-MiniLM-L6-v2")
            embeds = embed_chunks(chunks, st.session_state["embed_model"])
            st.session_state["rag_chunks"] = chunks
            st.session_state["rag_embeds"] = embeds
            st.session_state["rag_index"] = build_faiss_index(np.array(embeds))
            st.success(f"File loaded and indexed for RAG: {len(chunks)} chunks.")
        else:
            st.session_state["rag_chunks"] = None
            st.session_state["rag_embeds"] = None
            st.session_state["rag_index"] = None

    # --- Chat input and response logic ---
    prompt = st.chat_input("What is up?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # RAG: Retrieve context if knowledge base is loaded
        context = ""
        if st.session_state.get("rag_index") is not None:
            top_chunks = search_index(prompt, st.session_state["embed_model"], st.session_state["rag_index"], st.session_state["rag_chunks"], top_k=3)
            context = "\n".join(top_chunks)
        # Compose system prompt for LLM
        system_prompt = "You are a helpful assistant. Use the following context to answer the user's question.\n" + context
        messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        ]
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True,
        )
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Add Telekom HR branding colors
TELEKOM_PINK = "#e20074"
TELEKOM_DARK = "#22223b"
TELEKOM_LIGHT = "#f7f7fa"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {TELEKOM_LIGHT};
    }}
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1v0mbdj h1 {{
        color: {TELEKOM_PINK} !important;
    }}
    .st-emotion-cache-1v0mbdj h1 {{
        font-weight: 800;
        letter-spacing: 1px;
    }}
    .st-emotion-cache-1v0mbdj p {{
        color: {TELEKOM_DARK};
    }}
    .stChatMessage--user {{
        background: {TELEKOM_PINK}10 !important;
        border-left: 4px solid {TELEKOM_PINK} !important;
    }}
    .stChatMessage--assistant {{
        background: white !important;
        border-left: 4px solid {TELEKOM_DARK} !important;
    }}
    .stButton>button {{
        background-color: {TELEKOM_PINK} !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        font-weight: 600;
    }}
    .stSidebar {{
        background-color: {TELEKOM_PINK}10 !important;
    }}
    </style>
""", unsafe_allow_html=True)
