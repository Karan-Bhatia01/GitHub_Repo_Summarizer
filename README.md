# 📖 GitHub Repository Q&A System

Welcome to the **GitHub Repository Q&A System**, a powerful Streamlit web application designed to query the contents of any public GitHub repository. Leveraging advanced natural language processing, this tool fetches repository files, processes them intelligently, and delivers precise answers to your questions in a green IBM Plex Mono output box.

## 🌟 Features

- **Query Public Repositories**: Input any public GitHub repository URL and ask questions about its contents.
- **Intelligent File Processing**: Automatically distinguishes between text (e.g., `.md`, `.txt`) and code files (e.g., `.py`, `.js`) for optimized analysis.
- **Semantic Chunking**: Employs `SemanticChunker` for text files and `RecursiveCharacterTextSplitter` for code files to create meaningful document chunks.
- **Efficient Search**: Utilizes FAISS vector store for fast retrieval of relevant content based on your query.
- **AI-Powered Responses**: Integrates HuggingFace’s `Mixtral-8x7B-Instruct-v0.1` LLM to generate accurate and concise answers.
- **Robust Error Handling**: Validates repository URLs and GitHub tokens, providing clear error messages for easy troubleshooting.

## 🚀 Getting Started

Follow these steps to set up and run the GitHub Repository Q&A System locally.

### Prerequisites

- Python 3.8 or higher
- GitHub personal access token with `public_repo` scope
- HuggingFace API token
- A virtual environment (recommended)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Karan-Bhatia01/GitHub_Repo_Summarizer.git
   cd GitHub_Repo_Summarizer

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the project root and add your tokens:
   ```env
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
   GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token
   ```
   - **GitHub Token**: Generate at [GitHub > Settings > Developer settings > Personal access tokens > Tokens (classic)](https://github.com/settings/tokens), selecting `public_repo` scope.
   - **HuggingFace Token**: Obtain from your [HuggingFace account settings](https://huggingface.co/settings/tokens).

### Running the App

1. **Start the Streamlit App**
   ```bash
   streamlit run main.py
   ```
   - Open the URL provided in your browser (e.g., `http://localhost:8501`).

2. **Interact with the App**
   - Enter a public GitHub repository URL (e.g., `https://github.com/owner/repo` or `owner/repo`).
   - Type a query about the repository’s contents (e.g., "What are the main features of this project?").
   - Click **Get Answer** to view the response in a green IBM Plex Mono box.

## 📂 Project Structure

```
GitHub_Repo_Summarizer/
├── main.py                # Streamlit app and core logic
├── document_loader.py     # Loads and processes GitHub repository files
├── text_splitter.py       # Splits text and code into chunks
├── .env                   # Environment variables (not tracked)
├── README.md              # Project documentation
├── requirements.txt       # List of dependencies
```

## 🛠️ How It Works

1. **Repository Loading**:
   - Uses `PyGithub` to fetch all files from a public GitHub repository, dynamically detecting the default branch.
   - Validates the GitHub token for `public_repo` access.

2. **File Processing**:
   - Separates text and code files based on file extensions.
   - Splits text using `SemanticChunker` for natural language understanding and code using `RecursiveCharacterTextSplitter` for syntax-aware chunking.

3. **Vector Store Creation**:
   - Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2` and stores them in a FAISS vector store for fast retrieval.

4. **Query Answering**:
   - Retrieves relevant chunks using FAISS retriever.
   - Processes the query and context with HuggingFace’s LLM to generate a precise answer.
```
