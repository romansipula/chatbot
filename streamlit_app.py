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

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

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
            st.success("File loaded successfully. (Chunking and embedding logic goes here.)")

    # --- Chat input and response logic ---
    prompt = st.chat_input("What is up?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Dummy response for now (replace with RAG/LLM logic as needed)
        with st.chat_message("assistant"):
            st.markdown("I'm a bot! (RAG/LLM response goes here)")
        st.session_state.messages.append({"role": "assistant", "content": "I'm a bot! (RAG/LLM response goes here)"})
