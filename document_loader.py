from langchain_core.documents import Document
from github import Github, GithubException
from dotenv import load_dotenv
import os
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        """Initialize with optional GitHub token from environment variables."""
        load_dotenv()
        self.access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        self.github = Github(self.access_token) if self.access_token else Github()
        logger.info("GitHub client initialized.")

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

    def _validate_repo(self, repo_name):
        """Check if the repository exists and is accessible."""
        try:
            repo = self.github.get_repo(repo_name)
            logger.info(f"Repository {repo_name} is accessible.")
            return repo
        except GithubException as e:
            logger.error(f"Failed to access repository {repo_name}: {str(e)}")
            if e.status == 404:
                raise ValueError(f"Repository {repo_name} not found or inaccessible.")
            elif e.status == 401:
                raise ValueError("Invalid or insufficient GitHub token permissions.")
            raise ValueError(f"Error accessing repository: {str(e)}")

    def load_files(self, repo_url, branch=None):
        """Load all files from the repository and return as LangChain Documents."""
        try:
            repo_name = self.parse_github_repo(repo_url)
            repo = self._validate_repo(repo_name)
            
            # Get default branch if none provided
            if branch is None:
                branch = repo.default_branch
                logger.info(f"Using default branch: {branch}")

            # Get repository contents
            contents = repo.get_contents("", ref=branch)
            if not contents:
                logger.warning(f"No files found in {repo_name} on branch {branch}.")
                return []

            documents = []
            # Recursively fetch all files
            self._fetch_files(repo, "", branch, documents)
            
            if not documents:
                logger.warning(f"No valid files loaded from {repo_name}.")
                return []
            
            logger.info(f"Loaded {len(documents)} files from {repo_name} on branch {branch}")
            return documents

        except Exception as e:
            logger.error(f"Error loading files from {repo_url}: {str(e)}")
            raise ValueError(f"Failed to load files: {str(e)}")

    def _fetch_files(self, repo, path, branch, documents):
        """Recursively fetch all files from the repository."""
        try:
            contents = repo.get_contents(path, ref=branch)
            for content in contents:
                if content.type == "dir":
                    # Recurse into directories
                    self._fetch_files(repo, content.path, branch, documents)
                elif content.type == "file":
                    try:
                        # Skip non-text files (e.g., images) based on extension
                        if content.path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.zip', '.pdf')):
                            logger.debug(f"Skipping non-text file: {content.path}")
                            continue
                        file_content = content.decoded_content.decode("utf-8", errors="ignore")
                        doc = Document(
                            page_content=file_content,
                            metadata={"source": content.path, "repo": repo.full_name, "branch": branch}
                        )
                        documents.append(doc)
                        logger.debug(f"Loaded file: {content.path}")
                    except Exception as e:
                        logger.warning(f"Failed to load file {content.path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching contents from {path}: {str(e)}")

    def print_file_names(self, docs):
        """Print file names from loaded documents."""
        if not docs:
            logger.info("No files found in the repository.")
            return
        logger.info(f"Loaded {len(docs)} files:")
        for i, doc in enumerate(docs):
            file_name = doc.metadata.get('source', 'Unknown')
            file_name = file_name.split('/')[-1]
            logger.info(f"File {i+1}: {file_name}")

    def separate_files(self, docs):
        """Separate text files and code files based on file extensions."""
        text_extensions = {'.md', '.txt', '.rst'}
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.ts', '.html', '.css', '.sh', '.sql'}

        text_files = []
        code_files = []

        for doc in docs:
            file_name = doc.metadata.get('source', '')
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in text_extensions:
                text_files.append(doc)
            elif file_ext in code_extensions:
                code_files.append(doc)

        logger.info(f"Separated {len(text_files)} text files and {len(code_files)} code files")
        return text_files, code_files