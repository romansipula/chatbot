import streamlit as st
from openai import OpenAI
import os
import tempfile
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import docx2txt
from rag_utils import load_txt, load_pdf, load_docx, chunk_text
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Ensure Telekom HR CSS theme is applied globally
with open("telekom_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Professional HR Chatbot UI
st.set_page_config(page_title="HR Support Chatbot", page_icon="💼", layout="centered")

# Load configuration
def get_secret_value(key):
    """
    Get secret value from Streamlit Cloud secrets or environment variables.
    Streamlit Cloud typically provides lowercase keys, so we check that first.
    """
    # Try lowercase first (Streamlit Cloud format), then uppercase
    keys_to_try = [key.lower(), key]
    
    # Only try to access secrets if they exist (avoid StreamlitSecretNotFoundError)
    try:
        # Check if secrets are available at all
        if hasattr(st, 'secrets'):
            for test_key in keys_to_try:
                try:
                    # Try attribute access first (most common)
                    if hasattr(st.secrets, test_key):
                        value = getattr(st.secrets, test_key, None)
                        if value and str(value).strip():
                            return str(value).strip()
                    
                    # Try dict-style access
                    if hasattr(st.secrets, 'get'):
                        value = st.secrets.get(test_key)
                        if value and str(value).strip():
                            return str(value).strip()
                except (KeyError, AttributeError):
                    continue
    except Exception:
        # If secrets are not available at all, just continue to env variables
        pass
    
    # If not found in secrets, try environment variables
    env_value = os.getenv(key) or os.getenv(key.lower())
    if env_value:
        return env_value.strip()
    
    return ""

st.markdown("""
# 💼 HR Support Chatbot
Welcome to your HR assistant. Ask any questions about employee benefits, policies, leave, payroll, or other HR topics. Upload your HR handbook or policy documents in the sidebar to get answers based on your company's own information.
""")

# DEBUG: Let's see what's actually available in secrets
st.markdown("## 🔍 DEBUG: Secrets Detection")
try:
    st.write("Secrets available:", hasattr(st, 'secrets'))
    if hasattr(st, 'secrets'):
        try:
            secrets_dict = dict(st.secrets)
            st.write("Available secret keys:", list(secrets_dict.keys()))
            st.write("Number of secrets:", len(secrets_dict))
            
            # Show what each key contains (masked for security)
            for key in secrets_dict.keys():
                value = secrets_dict[key]
                if isinstance(value, str) and len(value) > 8:
                    st.write(f"- {key}: {value[:8]}...")
                else:
                    st.write(f"- {key}: {value}")
        except Exception as e:
            st.write(f"Error reading secrets dict: {e}")
            
        # Test our function
        st.write("Testing get_secret_value:")
        for key in ["OPENAI_API_KEY", "openai_api_key", "PINECONE_API_KEY", "pinecone_api_key"]:
            result = get_secret_value(key)
            st.write(f"- get_secret_value('{key}'): {'✅ Found' if result else '❌ Not found'}")
            
except Exception as e:
    st.write(f"Error accessing secrets: {e}")

st.markdown("---")

# Try to get OpenAI API key from Streamlit secrets first
openai_api_key = get_secret_value("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.warning("⚠️ OpenAI API key not found in secrets or environment variables")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()
else:
    st.success(f"✅ OpenAI API key loaded: {openai_api_key[:8]}...")

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Load credentials from Streamlit secrets or environment
    pinecone_api_key = get_secret_value("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
    pinecone_env     = get_secret_value("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index   = get_secret_value("PINECONE_INDEX_NAME") or os.getenv("PINECONE_INDEX_NAME")

    if not pinecone_api_key or not pinecone_env or not pinecone_index:
        st.error("❌ Pinecone credentials not found. Please configure your secrets in Streamlit Cloud.")
        st.markdown("### 📋 How to Configure Secrets in Streamlit Cloud:")
        st.markdown("""
        **Add these secrets in your Streamlit Cloud app settings:**

        ```toml
        openai_api_key = "your-openai-api-key"
        pinecone_api_key = "your-pinecone-api-key"
        pinecone_environment = "your-pinecone-environment"
        pinecone_index_name = "your-pinecone-index-name"
        ```

        **Steps:**
        1. Go to your Streamlit Cloud app
        2. Click on the gear icon (⚙️) for settings
        3. Go to the "Secrets" tab
        4. Add the above configuration with your actual values
        """)
        st.stop()

    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import OpenAIEmbeddings

    # Initialize Pinecone client and index
    pc = Pinecone(api_key=pinecone_api_key)
    if pinecone_index not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=pinecone_env)
        )
    index = pc.Index(pinecone_index)

    # Connect LangChain vector store
    try:
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=OpenAIEmbeddings(api_key=openai_api_key),
            namespace="employees"
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize vector store: {e}")
        st.stop()

    def get_context_from_pinecone(query: str, top_k: int = 3) -> str:
        """Retrieve and concatenate the top_k matching document chunks from Pinecone."""
        docs = vectorstore.similarity_search(query, k=top_k)
        return "\n\n".join([doc.page_content for doc in docs])

    def populate_pinecone_if_empty():
        """Populate Pinecone with default HR data if the index is empty."""
        try:
            # Check if there are any documents in the vectorstore
            test_docs = vectorstore.similarity_search("test", k=1)
            if not test_docs:
                # Index is empty, populate with default HR data
                with open("default_hr_data.txt", "r", encoding="utf-8") as f:
                    default_text = f.read()
                
                chunks = chunk_text(default_text)
                if chunks:
                    # Add documents to Pinecone
                    from langchain.schema import Document
                    documents = [Document(page_content=chunk) for chunk in chunks]
                    vectorstore.add_documents(documents)
                    st.sidebar.success(f"✅ Populated Pinecone with {len(chunks)} HR document chunks")
                    return True
            return False
        except Exception as e:
            st.sidebar.error(f"❌ Error populating Pinecone: {e}")
            return False

    # Populate Pinecone with default data if empty
    if st.sidebar.button("Populate Pinecone with Default HR Data"):
        populate_pinecone_if_empty()
    
    # Clear Pinecone index
    if st.sidebar.button("Clear Pinecone Index", type="secondary"):
        try:
            # Get the raw Pinecone index
            index = pc.Index(pinecone_index)
            # Delete all vectors in the namespace
            index.delete(delete_all=True, namespace="employees")
            st.sidebar.success("✅ Pinecone index cleared successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Error clearing Pinecone index: {e}")

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
    
    # Pinecone connection health check
    if st.sidebar.button("Test Pinecone Connection"):
        try:
            results = vectorstore.similarity_search("test connection", k=1)
            if results:
                st.sidebar.success(f"✅ Pinecone OK: found {len(results)} documents")
            else:
                st.sidebar.warning("⚠️ Pinecone OK but no documents found")
        except Exception as e:
            st.sidebar.error(f"❌ Pinecone connection failed: {e}")
    
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
            # File loaded successfully - Store chunks in Pinecone
            chunks = chunk_text(file_text)
            if chunks:
                from langchain.schema import Document
                documents = [Document(page_content=chunk) for chunk in chunks]
                try:
                    vectorstore.add_documents(documents)
                    st.success(f"✅ File uploaded and stored in Pinecone: {len(chunks)} chunks added.")
                except Exception as e:
                    st.error(f"❌ Error storing file in Pinecone: {e}")
            else:
                st.error("No content chunks extracted from the file.")
        else:
            st.error("Failed to load file content.")
    else:
        # Default HR data - using Pinecone for retrieval
        with open("default_hr_data.txt", "r", encoding="utf-8") as f:
            file_text = f.read()
        chunks = chunk_text(file_text)
        # TODO: Store default chunks in Pinecone vectorstore
        # For now, just show info message
        st.info("Using Pinecone for HR data retrieval. Upload your own file to add custom content.")

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
        # Always define employee_context
        employee_context = ""
        discount_keywords = ["bicycle discount", "bike discount", "discount for bicycle", "discount for bike", "bike benefit", "bicycle benefit"]
        # --- Extract user identity from prompt ---
        # Expanded regex to match: my name is, i am, i'm, im, this is
        user_name_match = re.search(r"(?:my name is|i am|i'm|im|this is)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)", prompt, re.IGNORECASE)
        user_first = user_last = None
        user_is_hr = False
        if user_name_match:
            user_first, user_last = user_name_match.group(1), user_name_match.group(2)
            user_row = emp_db[(emp_db["FirstName"]==user_first) & (emp_db["LastName"]==user_last)]
            if not user_row.empty:
                user_info = user_row.iloc[0]
                user_is_hr = "hr" in user_info.Position.lower()
        # --- Discount logic: always require user's name, enforce privacy strictly ---
        if any(kw in prompt.lower() for kw in discount_keywords):
            # Always require user's name (expanded regex)
            user_name_match = re.search(r"(?:my name is|i am|i'm|im|this is)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)", prompt, re.IGNORECASE)
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
            # Only allow discount for self unless user is HR
            if not user_is_hr:
                target_first, target_last = user_first, user_last
                # If prompt mentions another name, block and warn
                target_name_match = re.search(r"(?:for|about|of)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)", prompt)
                if target_name_match:
                    other_first, other_last = target_name_match.group(1), target_name_match.group(2)
                    if (other_first != user_first or other_last != user_last):
                        st.warning("You are only allowed to view your own discount. Please ask about yourself, or contact HR for information about others.")
                        st.stop()
            else:
                # HR can specify another name, but must use: for/about/of Firstname Lastname
                target_name_match = re.search(r"(?:for|about|of)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)", prompt)
                if target_name_match:
                    target_first, target_last = target_name_match.group(1), target_name_match.group(2)
                else:
                    target_first, target_last = user_first, user_last
            target_row = emp_db[(emp_db["FirstName"]==target_first) & (emp_db["LastName"]==target_last)]
            if target_row.empty:
                st.warning(f"Sorry, we could not find an employee named {target_first} {target_last} in our records. Please check the name or contact HR.")
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
        # --- Employee info extraction ---
        name_matches = re.findall(r"([A-Z][a-z]+)\s+([A-Z][a-z]+)", prompt)
        shown_employees = set()
        for first, last in name_matches:
            row = emp_db[(emp_db["FirstName"]==first) & (emp_db["LastName"]==last)]
            if not row.empty and (first, last) not in shown_employees:
                # Only block if user is not HR and is asking about someone else (not self)
                if not user_is_hr and user_first and user_last and (user_first != first or user_last != last):
                    st.warning("You are only allowed to view your own information. Please ask about yourself, or contact HR for information about others.")
                    st.stop()
                # Only show info if user is HR or is asking about self
                if user_is_hr or (user_first == first and user_last == last):
                    emp_info = row.iloc[0]
                    employee_context += f"Employee Info for {emp_info.FirstName} {emp_info.LastName}:\n- DOB: {emp_info.DOB}\n- First Day: {emp_info.FirstDay}\n- Position: {emp_info.Position}\n"
                    st.markdown(f"**Employee Info:**\n- Name: {emp_info.FirstName} {emp_info.LastName}\n- DOB: {emp_info.DOB}\n- First Day: {emp_info.FirstDay}\n- Position: {emp_info.Position}")
                shown_employees.add((first, last))
        # RAG: Retrieve context using Pinecone
        context = ""
        user_name = None
        name_match = re.search(r"(?:my name is|i am|this is)\s+(\w+)", prompt, re.IGNORECASE)
        if name_match:
            user_name = name_match.group(1).capitalize()
        
        # Retrieve relevant context from Pinecone
        user_question = prompt
        context = get_context_from_pinecone(user_question, top_k=3)
        # Compose system prompt for LLM, now with employee info if found
        system_prompt = f"You are a helpful assistant. Use the following context to answer the user's question. Only reveal the bicycle discount for the user's name ({user_name if user_name else 'unknown'}), and do not reveal discounts for any other names.\n" + employee_context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": f"Context:\n{context}"},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        ]
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True,
        )
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
