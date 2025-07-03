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

# Ensure Telekom HR CSS theme is applied globally
with open("telekom_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

    # Load default RAG data if no file is uploaded
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
    else:
        with open("default_hr_data.txt", "r", encoding="utf-8") as f:
            file_text = f.read()
        chunks = chunk_text(file_text)
        if "embed_model" not in st.session_state:
            st.session_state["embed_model"] = SentenceTransformer("all-MiniLM-L6-v2")
        embeds = embed_chunks(chunks, st.session_state["embed_model"])
        st.session_state["rag_chunks"] = chunks
        st.session_state["rag_embeds"] = embeds
        st.session_state["rag_index"] = build_faiss_index(np.array(embeds))
        st.info("Default HR data loaded for RAG. Upload your own file to override.")

    # Load mock employee DB into session state
    @st.cache_data
    def load_employee_db():
        df = pd.read_csv("mock_employee_db.csv", comment="#", names=["FirstName","LastName","DOB","FirstDay","Position"], skiprows=4)
        return df
    st.session_state["employee_db"] = load_employee_db()

    # --- Chat input and response logic ---
    prompt = st.chat_input("What is up?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Employee DB lookup logic
        import re
        emp_db = st.session_state["employee_db"]
        emp_match = re.search(r"(?:employee|person|colleague|myself|me|name is|who is)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)", prompt)
        if emp_match:
            first, last = emp_match.group(1), emp_match.group(2)
            row = emp_db[(emp_db["FirstName"]==first) & (emp_db["LastName"]==last)]
            if not row.empty:
                emp_info = row.iloc[0]
                st.markdown(f"**Employee Info:**\n- Name: {emp_info.FirstName} {emp_info.LastName}\n- DOB: {emp_info.DOB}\n- First Day: {emp_info.FirstDay}\n- Position: {emp_info.Position}")
        # RAG: Retrieve context if knowledge base is loaded
        context = ""
        # Custom logic: filter discount info for privacy
        user_name = None
        # Try to extract the user's name from the prompt (e.g., "My name is John")
        name_match = re.search(r"(?:my name is|i am|this is)\s+(\w+)", prompt, re.IGNORECASE)
        if name_match:
            user_name = name_match.group(1).capitalize()
        # If asking about bicycle discount and a name is detected, filter context
        if (
            st.session_state.get("rag_index") is not None and
            ("bicycle" in prompt.lower() or "discount" in prompt.lower()) and
            user_name
        ):
            # Only include the discount line for the user's name
            filtered_chunks = []
            for chunk in st.session_state["rag_chunks"]:
                lines = chunk.split("\n")
                filtered = []
                for line in lines:
                    # Only allow the discount line for the user's name, and always allow non-discount lines
                    if user_name in line and "% off" in line:
                        filtered.append(line)
                    elif "% off" in line:
                        continue  # skip all other discount lines
                    else:
                        filtered.append(line)
                filtered_chunks.append("\n".join(filtered))
            top_chunks = search_index(prompt, st.session_state["embed_model"], st.session_state["rag_index"], filtered_chunks, top_k=3)
            context = "\n".join(top_chunks)
        elif st.session_state.get("rag_index") is not None:
            top_chunks = search_index(prompt, st.session_state["embed_model"], st.session_state["rag_index"], st.session_state["rag_chunks"], top_k=3)
            context = "\n".join(top_chunks)
        # Compose system prompt for LLM
        system_prompt = f"You are a helpful assistant. Use the following context to answer the user's question. Only reveal the bicycle discount for the user's name ({user_name if user_name else 'unknown'}), and do not reveal discounts for any other names.\n" + context
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
