from langchain_community.document_loaders.github import GithubFileLoader
from dotenv import load_dotenv
import os

class DocumentLoader:
    def __init__(self):
        """Initialize with environment variables."""
        load_dotenv()
        self.access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN not found in .env file")

    def parse_github_repo(self, url):
        """Parse repo URL to owner/repo format."""
        url = url.strip()
        if url.startswith("https://github.com/"):
            repo_path = url[len("https://github.com/"):].rstrip("/")
            if repo_path.count("/") == 1:
                return repo_path
        if url.count("/") == 1 and not url.startswith("http"):
            return url
        raise ValueError("Invalid format. Use 'owner/repo' or 'https://github.com/owner/repo'")

    def load_files(self, repo_url, branch="main"):
        """Load all files from the repository and return documents."""
        try:
            repo = self.parse_github_repo(repo_url)
            loader = GithubFileLoader(
                repo=repo,
                access_token=self.access_token,
                branch=branch,
                file_filter=lambda file_path: True  # Load all files
            )
            docs = loader.load()
            return docs
        except Exception as e:
            print(f"Error loading files: {e}")
            return []

    def print_file_names(self, docs):
        """Print file names from loaded documents."""
        print(f"Loaded {len(docs)} files:")
        if not docs:
            print("No files found in the repository.")
            return
        for i, doc in enumerate(docs):
            # Try file_path, fallback to source
            file_name = doc.metadata.get('file_path', doc.metadata.get('source', 'Unknown'))
            if file_name != 'Unknown' and 'source' in doc.metadata:
                file_name = file_name.split('/')[-1]  # Extract file name from URL
            print(f"File {i+1}: {file_name}")

    def separate_files(self, docs):
        """Separate text files and code files based on file extensions."""
        text_extensions = {'.md', '.txt', '.rst', '.doc', '.docx', '.pdf'}
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.ts', '.html', '.css', '.sh', '.sql'}

        text_files = []
        code_files = []

        for doc in docs:
            file_name = doc.metadata.get('file_path', doc.metadata.get('source', ''))
            file_ext = os.path.splitext(file_name)[1].lower()

            if file_ext in text_extensions:
                text_files.append(doc)
            elif file_ext in code_extensions:
                code_files.append(doc)

        return text_files, code_files
