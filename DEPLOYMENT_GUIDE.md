# HR Chatbot Deployment Guide

## Quick Start Guide

### 1. Local Development Setup

#### Prerequisites
- Python 3.9+
- Git
- OpenAI API key
- Pinecone API key and index

#### Installation
```bash
# Clone the repository
git clone https://github.com/romansipula/chatbot.git
cd chatbot

# Create virtual environment
python -m venv chatbot_env
# On Windows
chatbot_env\Scripts\activate
# On macOS/Linux
source chatbot_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Configuration
Create a `.streamlit/secrets.toml` file:
```toml
openai_api_key = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "your-pinecone-index-name"
```

#### Run Locally
```bash
streamlit run streamlit_app.py
```

### 2. Streamlit Cloud Deployment

#### Steps
1. Fork this repository to your GitHub account
2. Sign up for [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub account
4. Deploy the app by selecting your forked repository
5. Configure secrets in Streamlit Cloud:
   - Go to your app settings
   - Add secrets in the TOML format shown above

#### Streamlit Cloud Secrets
```toml
openai_api_key = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "your-pinecone-index-name"
```

### 3. Pinecone Setup

#### Create Pinecone Index
1. Sign up at [Pinecone](https://www.pinecone.io/)
2. Create a new index with these settings:
   - **Name**: Choose any name (e.g., "hr-chatbot")
   - **Dimension**: 1536 (for OpenAI embeddings)
   - **Metric**: cosine
   - **Cloud**: AWS or GCP (your choice)
   - **Region**: Choose closest to your users

#### Initial Data Population
1. Run the application
2. Click "Populate Pinecone with Default HR Data" in the sidebar
3. Upload your own HR documents using the file uploader

### 4. Testing the Application

#### Basic Tests
1. **Connection Test**: Click "Test Pinecone Connection"
2. **Data Population**: Click "Populate Pinecone with Default HR Data"
3. **Employee Query**: Type "My name is John Smith" and ask about benefits
4. **Discount Calculation**: Ask "What is my bicycle discount? My name is John Smith"

#### Sample Queries
- "What are the company's vacation policies?"
- "My name is Sarah Johnson, what is my bicycle discount?"
- "Tell me about health insurance benefits"
- "How do I request time off?"

### 5. Troubleshooting

#### Common Issues
1. **API Key Errors**: Ensure your OpenAI and Pinecone keys are correct
2. **Index Not Found**: Verify your Pinecone index name is correct
3. **No Documents Found**: Run the data population function first
4. **Employee Not Found**: Check the employee database format

#### Debug Steps
1. Check the sidebar for connection status
2. Use the "Test Pinecone Connection" button
3. Verify your API keys in the secrets configuration
4. Check the terminal/logs for detailed error messages

### 6. Customization

#### Adding Your Own Data
1. Replace `default_hr_data.txt` with your company's HR information
2. Update `mock_employee_db.csv` with your employee data
3. Modify the CSS in `telekom_theme.css` for your branding

#### Employee Database Format
The CSV should have these columns:
- FirstName
- LastName
- DOB (Date of Birth)
- FirstDay (Start Date)
- Position

### 7. Security Considerations

#### Data Privacy
- Employee information is protected by role-based access
- Only HR personnel can access other employees' data
- Regular employees can only view their own information

#### API Security
- Never commit API keys to version control
- Use environment variables or Streamlit secrets
- Rotate API keys regularly

### 8. Performance Optimization

#### Pinecone Best Practices
- Use appropriate chunk sizes (current: ~1000 characters)
- Regularly clean up old/unused vectors
- Monitor your Pinecone usage and costs

#### Streamlit Optimization
- Use `@st.cache_data` for expensive operations
- Minimize API calls in the main loop
- Consider using Streamlit's session state for user data

### 9. Support and Maintenance

#### Regular Tasks
- Monitor API usage and costs
- Update employee database regularly
- Refresh HR documentation in Pinecone
- Review and update the knowledge base

#### Monitoring
- Check Streamlit Cloud logs for errors
- Monitor Pinecone query performance
- Track user engagement and common queries

## Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review the application logs
3. Verify your API keys and configuration
4. Test the connection functions in the sidebar
