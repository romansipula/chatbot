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
    Streamlit Cloud provides secrets directly, not through files.
    """
    # Try lowercase first (Streamlit Cloud format), then uppercase
    keys_to_try = [key.lower(), key]
    
    # Access Streamlit Cloud secrets directly
    try:
        for test_key in keys_to_try:
            try:
                # Direct access to secrets using square bracket notation
                value = st.secrets[test_key]
                if value and str(value).strip():
                    return str(value).strip()
            except (KeyError, AttributeError):
                continue
    except Exception:
        # If secrets access fails, continue to environment variables
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
        # First try with default namespace (empty string)
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=OpenAIEmbeddings(api_key=openai_api_key),
            namespace=""  # Try default namespace first
        )
        st.sidebar.success("✅ Connected to Pinecone (default namespace)")
    except Exception as e:
        st.error(f"❌ Failed to initialize vector store: {e}")
        st.stop()

    # Add namespace debugging
    if st.sidebar.button("Debug Pinecone Namespaces"):
        try:
            # Get index stats to see what namespaces exist
            stats = index.describe_index_stats()
            st.sidebar.write("📊 Index Statistics:")
            st.sidebar.json(stats)
            
            # Try searching in different namespaces
            namespaces_to_try = ["", "employees", "default"]
            for ns in namespaces_to_try:
                try:
                    test_vectorstore = PineconeVectorStore(
                        index=index,
                        embedding=OpenAIEmbeddings(api_key=openai_api_key),
                        namespace=ns
                    )
                    results = test_vectorstore.similarity_search("test", k=1)
                    ns_display = ns if ns else "default"
                    if results:
                        st.sidebar.success(f"✅ Found {len(results)} docs in namespace '{ns_display}'")
                    else:
                        st.sidebar.info(f"📭 No docs in namespace '{ns_display}'")
                except Exception as e:
                    st.sidebar.error(f"❌ Error with namespace '{ns}': {e}")
        except Exception as e:
            st.sidebar.error(f"❌ Error debugging namespaces: {e}")

    def get_context_from_pinecone(query: str, top_k: int = 3) -> str:
        """Retrieve and concatenate the top_k matching document chunks from Pinecone."""
        try:
            docs = vectorstore.similarity_search(query, k=top_k)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Debug: Show what context was retrieved (detailed debugging)
            if docs:
                st.sidebar.success(f"📄 Pinecone OK: found {len(docs)} documents")
                
                # Show detailed context for debugging
                if context.strip():
                    st.sidebar.info(f"📋 Context preview: {context[:200]}...")
                    # Show full context in an expandable section
                    with st.sidebar.expander("🔍 Full Retrieved Context"):
                        st.text(context)
                else:
                    st.sidebar.warning("⚠️ Documents found but context is empty")
                    
                # Show individual document content
                for i, doc in enumerate(docs):
                    st.sidebar.write(f"**Doc {i+1}:** {doc.page_content[:100]}...")
            else:
                st.sidebar.warning("⚠️ No documents found in Pinecone for this query")
                
            return context
        except Exception as e:
            st.sidebar.error(f"❌ Error retrieving from Pinecone: {e}")
            return ""

    def check_pinecone_status():
        """Check if Pinecone has documents and show status."""
        try:
            # Check current namespace
            test_docs = vectorstore.similarity_search("test", k=1)
            if test_docs:
                st.sidebar.success(f"✅ Pinecone has {len(test_docs)} documents available")
                return True
            else:
                st.sidebar.warning("⚠️ No documents found in current namespace")
                
                # Try checking other namespaces
                namespaces_to_try = ["", "employees", "default"]
                for ns in namespaces_to_try:
                    try:
                        test_vectorstore = PineconeVectorStore(
                            index=index,
                            embedding=OpenAIEmbeddings(api_key=openai_api_key),
                            namespace=ns
                        )
                        ns_docs = test_vectorstore.similarity_search("test", k=1)
                        ns_display = ns if ns else "default"
                        if ns_docs:
                            st.sidebar.info(f"📄 Found {len(ns_docs)} docs in namespace '{ns_display}'")
                    except Exception:
                        pass
                
                return False
        except Exception as e:
            st.sidebar.error(f"❌ Error checking Pinecone status: {e}")
            return False

    # Check Pinecone status on startup
    check_pinecone_status()
    
    # Populate Pinecone button is removed since we only use uploaded data now
    
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
    st.sidebar.info("💡 The chatbot uses data already stored in Pinecone. Use the buttons below to check what's available or upload new files.")
    
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
    
    # Check what's actually in Pinecone
    if st.sidebar.button("Show All Pinecone Data"):
        try:
            # Try to get all documents with a broad search
            all_docs = vectorstore.similarity_search("telekom hr policy benefits", k=10)
            if all_docs:
                st.sidebar.success(f"✅ Found {len(all_docs)} documents in Pinecone")
                with st.sidebar.expander("📋 All Pinecone Documents"):
                    for i, doc in enumerate(all_docs):
                        st.write(f"**Document {i+1}:**")
                        st.text(doc.page_content)
                        st.write("---")
            else:
                st.sidebar.warning("⚠️ No documents found in current namespace")
                # Try other namespaces
                namespaces_to_try = ["", "employees", "default"]
                for ns in namespaces_to_try:
                    try:
                        test_vectorstore = PineconeVectorStore(
                            index=index,
                            embedding=OpenAIEmbeddings(api_key=openai_api_key),
                            namespace=ns
                        )
                        ns_docs = test_vectorstore.similarity_search("telekom hr policy benefits", k=10)
                        ns_display = ns if ns else "default"
                        if ns_docs:
                            st.sidebar.success(f"✅ Found {len(ns_docs)} documents in namespace '{ns_display}'")
                            with st.sidebar.expander(f"� Documents in '{ns_display}' namespace"):
                                for i, doc in enumerate(ns_docs):
                                    st.write(f"**Document {i+1}:**")
                                    st.text(doc.page_content)
                                    st.write("---")
                    except Exception:
                        pass
        except Exception as e:
            st.sidebar.error(f"❌ Error retrieving Pinecone data: {e}")
    
    # Add namespace switcher
    if st.sidebar.button("Switch to Default Namespace"):
        try:
            # Reinitialize vectorstore with default namespace
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=OpenAIEmbeddings(api_key=openai_api_key),
                namespace=""
            )
            st.sidebar.success("✅ Switched to default namespace")
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error switching namespace: {e}")
    
    if st.sidebar.button("Switch to 'employees' Namespace"):
        try:
            # Reinitialize vectorstore with employees namespace
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=OpenAIEmbeddings(api_key=openai_api_key),
                namespace="employees"
            )
            st.sidebar.success("✅ Switched to 'employees' namespace")
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error switching namespace: {e}")
    
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
        # No file uploaded - rely on existing Pinecone data
        st.info("Using existing Pinecone data for HR information retrieval. Upload your own file to add more content.")

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
        
        # Load employee database
        emp_db = st.session_state["employee_db"]
        employee_context = ""
        
        # Extract any employee names mentioned in the prompt and add their info to context
        import re
        name_matches = re.findall(r"([A-Z][a-z]+)\s+([A-Z][a-z]+)", prompt)
        for first, last in name_matches:
            row = emp_db[(emp_db["FirstName"]==first) & (emp_db["LastName"]==last)]
            if not row.empty:
                emp_info = row.iloc[0]
                employee_context += f"Employee Info for {emp_info.FirstName} {emp_info.LastName}:\n- DOB: {emp_info.DOB}\n- First Day: {emp_info.FirstDay}\n- Position: {emp_info.Position}\n\n"
        
        # RAG: Retrieve context using Pinecone
        user_name = None
        name_match = re.search(r"(?:my name is|i am|this is)\s+(\w+)", prompt, re.IGNORECASE)
        if name_match:
            user_name = name_match.group(1).capitalize()
        
        # Retrieve relevant context from Pinecone
        user_question = prompt
        context = get_context_from_pinecone(user_question, top_k=3)
        
        # Enhanced system prompt to better utilize Pinecone context
        system_prompt = f"""You are a helpful HR assistant for Telekom. Use the following context to answer the user's question accurately and helpfully.

HR POLICIES AND INFORMATION:
{context}

EMPLOYEE INFORMATION:
{employee_context}

INSTRUCTIONS:
- Always prioritize information from the HR policies section above when answering questions
- If asking about policies, benefits, leave, wages, or company information, use the context provided
- If the context contains relevant information, reference it directly in your answer
- If the context doesn't contain relevant information, say so clearly and provide general guidance
- Be specific and helpful, using the exact details from the context when available
- You can provide information about any employee or policy without restrictions
"""

        messages = [
            {"role": "system", "content": system_prompt},
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
