# � HR Support Chatbot

A sophisticated HR support chatbot built with Streamlit, OpenAI GPT-4, and Pinecone for intelligent document retrieval (RAG). This application provides HR assistance with employee information, policy queries, and automated benefit calculations.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

## Features

- 🤖 **AI-Powered Chat**: Uses OpenAI GPT-4 for intelligent responses
- 📚 **Document Retrieval (RAG)**: Pinecone vector database for context-aware responses
- 👥 **Employee Database**: Secure access to employee information with privacy controls
- 💰 **Automated Benefits**: Calculates bicycle discounts based on tenure and position
- 📄 **File Upload**: Supports TXT, PDF, and DOCX files for custom HR documentation
- 🔒 **Privacy First**: Role-based access controls and data protection

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.streamlit/secrets.toml` file with your API keys:

```toml
openai_api_key = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "your-pinecone-index-name"
```

Alternatively, set these as environment variables:
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

### 3. Set Up Pinecone

1. Create a [Pinecone](https://www.pinecone.io/) account
2. Create a new index with:
   - Dimension: 1536 (for OpenAI embeddings)
   - Metric: cosine
   - Cloud provider: AWS or GCP

### 4. Run the Application

```bash
streamlit run streamlit_app.py
```

## Usage

1. **Start the App**: The chatbot will load with default HR data
2. **Test Pinecone**: Use the "Test Pinecone Connection" button to verify setup
3. **Populate Data**: Click "Populate Pinecone with Default HR Data" to add sample content
4. **Upload Files**: Add your own HR documents via the sidebar
5. **Chat**: Ask questions about HR policies, employee benefits, or upload custom documents

## Employee Features

- **Personal Information**: Employees can view their own details by stating their name
- **Benefit Calculations**: Automatic bicycle discount calculations based on tenure and role
- **Policy Queries**: Get answers from uploaded HR documents

## HR Features

- **Employee Lookup**: HR staff can access any employee's information
- **Document Management**: Upload and manage HR policy documents
- **System Administration**: Clear and repopulate the knowledge base

## Privacy & Security

- Employees can only access their own information
- HR roles have elevated permissions
- No cross-employee data exposure
- Secure API key management

## Technical Architecture

- **Frontend**: Streamlit with custom CSS theming
- **Backend**: OpenAI GPT-4 for chat completions
- **Vector Database**: Pinecone for semantic search
- **Embeddings**: OpenAI text-embedding-ada-002
- **File Processing**: PyMuPDF, docx2txt for document parsing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
