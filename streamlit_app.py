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
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
