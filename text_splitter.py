from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

class TextSplitter:
    def __init__(self):
        """Initialize with environment variables."""
        load_dotenv()
        self.hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.hf_api_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

    def text_file_splitting(self, text_files, use_semantic=True):
        """Split a list of text file documents using SemanticChunker or RecursiveCharacterTextSplitter."""
        try:
            # Initialize the appropriate splitter
            if use_semantic:
                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'token': self.hf_api_token}
                )
                text_splitter = SemanticChunker(
                    embedding_model,
                    breakpoint_threshold_type="standard_deviation",
                    breakpoint_threshold_amount=2
                )
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=200,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ".", " ", ""]
                )

            # Process each text file document
            all_chunks = []
            for doc in text_files:
                text = doc.page_content
                # Split the text and preserve metadata
                chunks = text_splitter.create_documents([text], metadatas=[doc.metadata])
                all_chunks.extend(chunks)

            return all_chunks, f"Split {len(text_files)} text files into {len(all_chunks)} chunks"

        except Exception as e:
            return [], f"Error splitting text files: {str(e)}"

    def code_file_splitting(self, code_files):
        """Split a list of code file documents using RecursiveCharacterTextSplitter for Python."""
        try:
            # Initialize the splitter for Python code
            code_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=300,
                chunk_overlap=50
            )

            # Process each code file document
            all_chunks = []
            for doc in code_files:
                text = doc.page_content
                # Split the code and preserve metadata
                chunks = code_splitter.create_documents([text], metadatas=[doc.metadata])
                all_chunks.extend(chunks)

            return all_chunks, f"Split {len(code_files)} code files into {len(all_chunks)} chunks"

        except Exception as e:
            return [], f"Error splitting code files: {str(e)}"

    def store_in_vectorstore(self, chunks, store_path="faiss_index"):
        """Store chunks in a FAISS vector store."""
        try:
            # Initialize embedding model
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'token': self.hf_api_token}
            )

            # Create FAISS vector store from chunks
            vector_store = FAISS.from_documents(chunks, embedding_model)

            # Save the vector store locally
            vector_store.save_local(store_path)

            return vector_store, f"Stored {len(chunks)} chunks in FAISS vector store at {store_path}"

        except Exception as e:
            return None, f"Error storing chunks in vector store: {str(e)}"