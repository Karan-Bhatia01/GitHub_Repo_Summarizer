import streamlit as st
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from prompts import create_prompt_templates, get_prompt_variables
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load environment variables
load_dotenv()
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

# Initialize Streamlit app
st.set_page_config(page_title="GitHub Repo Q&A", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
    .output-text {
    font-family: 'Fira Code', 'JetBrains Mono', 'IBM Plex Mono', monospace;
    font-size: 13.5px;
    white-space: pre-wrap;
    background-color: #1e1e1e;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-left: 4px solid #00c8a8;
    padding: 12px 16px;
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)
st.title("GitHub Repository Q&A System")

# Function to validate GitHub repo link
def validate_github_url(url):
    """Validate GitHub repository URL or owner/repo format."""
    url = url.strip()
    pattern = r"^(https://github\.com/)?([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)$"
    match = re.match(pattern, url)
    if match:
        return match.group(2)  # Return owner/repo
    return None

# Input fields
repo_url = st.text_input(
    "GitHub Repository URL",
    value=""
)
user_query = st.text_area("Enter your query about the repository", height=100)
submit_button = st.button("Get Answer")

if submit_button and repo_url and user_query:
    with st.spinner("Processing repository and query..."):
        try:
            # Validate and parse GitHub URL
            repo = validate_github_url(repo_url)
            if not repo:
                st.error("Invalid GitHub repository format. Use 'https://github.com/owner/repo' or 'owner/repo'.")
                st.stop()

            # Initialize loader and splitter
            loader = DocumentLoader()
            splitter = TextSplitter()

            # Load files from the GitHub repository
            docs = loader.load_files(repo)
            if not docs:
                st.error("No files loaded from the repository. Check the repository URL, ensure it contains files, or verify your GitHub token has 'public_repo' scope.")
                st.stop()

            # Separate text and code files
            text_files, code_files = loader.separate_files(docs)

            # Split text files
            text_chunks, text_message = splitter.text_file_splitting(text_files, use_semantic=True)

            # Split code files
            code_chunks, code_message = splitter.code_file_splitting(code_files)

            # Store all chunks in a FAISS vector store
            all_chunks = text_chunks + code_chunks
            vector_store, store_message = splitter.store_in_vectorstore(all_chunks, store_path="faiss_index")
            if not vector_store:
                st.error("Failed to create vector store.")
                st.stop()

            # Initialize FAISS retriever
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})

            # Initialize LLM
            llm = ChatGroq(
                model='llama-3.1-8b-instant',
                max_tokens=1024,
                temperature=1.5,
                timeout=None,
                max_retries=2,
            )

            # Create prompt template
            prompt = create_prompt_templates()

            # Build the chain with defensive chat_history access
            chain = (
                {
                    "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
                    "chat_history": lambda x: st.session_state.get("chat_history", []),
                    "query": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            # Get the answer
            answer = chain.invoke(user_query)

            # Update chat history
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(HumanMessage(content=answer))

            # Display the answer
            st.markdown(f"<div class='output-text'>**Answer:**\n{answer}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

else:
    if submit_button:
        st.warning("Please provide both a repository URL and a query.")