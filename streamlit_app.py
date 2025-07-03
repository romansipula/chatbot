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
        import re
        emp_db = st.session_state["employee_db"]
        # Improved: Find any Firstname Lastname in the prompt
        name_matches = re.findall(r"([A-Z][a-z]+)\s+([A-Z][a-z]+)", prompt)
        employee_context = ""
        shown_employees = set()
        for first, last in name_matches:
            row = emp_db[(emp_db["FirstName"]==first) & (emp_db["LastName"]==last)]
            if not row.empty and (first, last) not in shown_employees:
                emp_info = row.iloc[0]
                employee_context += f"Employee Info for {emp_info.FirstName} {emp_info.LastName}:\n- DOB: {emp_info.DOB}\n- First Day: {emp_info.FirstDay}\n- Position: {emp_info.Position}\n"
                st.markdown(f"**Employee Info:**\n- Name: {emp_info.FirstName} {emp_info.LastName}\n- DOB: {emp_info.DOB}\n- First Day: {emp_info.FirstDay}\n- Position: {emp_info.Position}")
                shown_employees.add((first, last))
        # --- Discount logic: ask for name if needed, privacy restriction ---
        discount_keywords = ["bicycle discount", "bike discount", "discount for bicycle", "discount for bike"]
        if any(kw in prompt.lower() for kw in discount_keywords):
            # Try to extract target name (the person discount is being asked for)
            target_name_match = re.search(r"(?:for|about|of)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)", prompt)
            # Try to extract user's own name
            user_name_match = re.search(r"(?:my name is|i am|this is)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)", prompt)
            # If not found, ask for user's name
            if not user_name_match:
                st.warning("To calculate your bicycle discount, please provide your full name (e.g., 'My name is John Smith').")
                st.stop()
            user_first, user_last = user_name_match.group(1), user_name_match.group(2)
            user_row = emp_db[(emp_db["FirstName"]==user_first) & (emp_db["LastName"]==user_last)]
            if user_row.empty:
                st.warning(f"Sorry, we could not find an employee named {user_first} {user_last} in our records. Please check your name or contact HR.")
                st.stop()
            user_info = user_row.iloc[0]
            user_is_hr = "hr" in user_info.Position.lower()
            # Determine whose discount is being asked for
            if target_name_match:
                target_first, target_last = target_name_match.group(1), target_name_match.group(2)
            else:
                target_first, target_last = user_first, user_last
            target_row = emp_db[(emp_db["FirstName"]==target_first) & (emp_db["LastName"]==target_last)]
            if target_row.empty:
                st.warning(f"Sorry, we could not find an employee named {target_first} {target_last} in our records. Please check the name or contact HR.")
                st.stop()
            # Privacy restriction: Only allow if user is HR or asking about themselves
            if not user_is_hr and (user_first != target_first or user_last != target_last):
                st.warning("For privacy reasons, you can only view your own bicycle discount. If you need information about another employee, please contact HR.")
                st.stop()
            emp_info = target_row.iloc[0]
            # Calculate years in company
            from datetime import datetime
            try:
                start_date = pd.to_datetime(emp_info.FirstDay)
                today = pd.Timestamp(datetime.now().date())
                years = (today - start_date).days // 365
            except Exception:
                years = 0
            # Determine base discount
            if years <= 2:
                base = 5
            elif years <= 4:
                base = 10
            elif years <= 6:
                base = 20
            elif years <= 10:
                base = 30
            else:
                base = 40
            # Job type bonuses
            position = emp_info.Position.lower()
            bonus = 0
            if any(x in position for x in ["it", "software", "engineer", "developer", "data", "network"]):
                bonus += 20
            if "hr" in position:
                bonus += 10
            if any(x in position for x in ["manager", "director", "lead", "head"]):
                bonus += 15
            total_discount = base + bonus
            if total_discount > 99:
                total_discount = 99
            st.success(f"Bicycle discount for {emp_info.FirstName} {emp_info.LastName}: {total_discount}% (base: {base}%, bonus: {bonus}% for position: {emp_info.Position}, years in company: {years}).")
            st.stop()
        # RAG: Retrieve context if knowledge base is loaded
        context = ""
        user_name = None
        name_match = re.search(r"(?:my name is|i am|this is)\s+(\w+)", prompt, re.IGNORECASE)
        if name_match:
            user_name = name_match.group(1).capitalize()
        if (
            st.session_state.get("rag_index") is not None and
            ("bicycle" in prompt.lower() or "discount" in prompt.lower()) and
            user_name
        ):
            filtered_chunks = []
            for chunk in st.session_state["rag_chunks"]:
                lines = chunk.split("\n")
                filtered = []
                for line in lines:
                    if user_name in line and "% off" in line:
                        filtered.append(line)
                    elif "% off" in line:
                        continue
                    else:
                        filtered.append(line)
                filtered_chunks.append("\n".join(filtered))
            top_chunks = search_index(prompt, st.session_state["embed_model"], st.session_state["rag_index"], filtered_chunks, top_k=3)
            context = "\n".join(top_chunks)
        elif st.session_state.get("rag_index") is not None:
            top_chunks = search_index(prompt, st.session_state["embed_model"], st.session_state["rag_index"], st.session_state["rag_chunks"], top_k=3)
            context = "\n".join(top_chunks)
        # Compose system prompt for LLM, now with employee info if found
        system_prompt = f"You are a helpful assistant. Use the following context to answer the user's question. Only reveal the bicycle discount for the user's name ({user_name if user_name else 'unknown'}), and do not reveal discounts for any other names.\n" + employee_context + context
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
