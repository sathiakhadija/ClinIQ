"""
NHS document ingestion pipeline.

Loads NHS PDFs, chunks them, embeds them with OpenAI embeddings,
and stores them in ChromaDB for retrieval.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables at module level
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def require_openai_api_key() -> None:
    """Fail early with a clear setup message before embedding calls."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your OpenAI API key."
        )


def load_nhs_documents(docs_dir: str = None) -> list:
    """
    Load all PDF files from the NHS documents directory.

    Args:
        docs_dir: Path to directory containing NHS PDFs.
                 Defaults to NHS_DOCS_DIR env var.

    Returns:
        List of LangChain Document objects with metadata.
        Each chunk will have metadata: source (filename), page number.

    Raises:
        FileNotFoundError: If directory does not exist.
        ValueError: If no PDF files are found in directory.

    Logs:
        - Number of PDFs found
        - Total document pages loaded
    """
    if docs_dir is None:
        docs_dir = os.getenv("NHS_DOCS_DIR", "./data/nhs_docs")

    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"NHS documents directory not found: {docs_dir}")

    # Find all PDF files
    pdf_files = list(docs_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {docs_dir}")

    logger.info(f"Found {len(pdf_files)} NHS PDF files")

    documents = []
    for pdf_file in pdf_files:
        logger.info(f"Loading {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        documents.extend(docs)
        logger.info(f"  Loaded {len(docs)} pages from {pdf_file.name}")

    logger.info(f"Total documents loaded: {len(documents)} pages")
    return documents


def chunk_documents(
    documents: list, chunk_size: int = None, chunk_overlap: int = None
) -> list:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Size of each chunk in tokens. Defaults to CHUNK_SIZE env var (500).
        chunk_overlap: Overlap between chunks in tokens. Defaults to CHUNK_OVERLAP env var (50).

    Returns:
        List of Document chunks.

    Logs:
        - Total chunks created
        - Average chunk size
    """
    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", "500"))

    if chunk_overlap is None:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

    logger.info(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")

    # RecursiveCharacterTextSplitter tries to split on these in order:
    # paragraphs, newlines, sentences, words, characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Calculate average chunk size
    if chunks:
        avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
    else:
        avg_size = 0

    logger.info(f"Created {len(chunks)} chunks (avg size: {avg_size:.0f} chars)")
    return chunks


def create_vector_store(chunks: list, persist_dir: str = None) -> Chroma:
    """
    Embed chunks using text-embedding-3-small and store in ChromaDB.

    Args:
        chunks: List of Document chunks to embed.
        persist_dir: Directory to persist ChromaDB. Defaults to CHROMA_PERSIST_DIR env var.

    Returns:
        Chroma vector store instance.

    Logs:
        - Embedding progress every 50 chunks
    """
    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    # Create persist directory if it doesn't exist
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    require_openai_api_key()

    logger.info("Embedding chunks with text-embedding-3-small...")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create vector store
    # Process in batches to log progress
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        logger.info(f"  Embedded {batch_end}/{len(chunks)} chunks")

    # Create ChromaDB vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="nhs_guidelines",
        persist_directory=persist_dir,
    )

    logger.info(f"Vector store created and persisted to {persist_dir}")
    return vector_store


def load_existing_vector_store(persist_dir: str = None) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        persist_dir: Directory containing ChromaDB. Defaults to CHROMA_PERSIST_DIR env var.

    Returns:
        Chroma vector store instance.

    Raises:
        FileNotFoundError: If persist directory does not exist.
    """
    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    persist_path = Path(persist_dir)

    if not persist_path.exists():
        raise FileNotFoundError(f"ChromaDB directory not found: {persist_dir}")

    logger.info(f"Loading existing vector store from {persist_dir}...")
    require_openai_api_key()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma(
        collection_name="nhs_guidelines",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    logger.info("Vector store loaded successfully")
    return vector_store


def run_ingestion_pipeline(docs_dir: str = None, persist_dir: str = None) -> dict:
    """
    Full ingestion pipeline: load → chunk → embed → store.

    Args:
        docs_dir: NHS documents directory. Defaults to NHS_DOCS_DIR env var.
        persist_dir: ChromaDB directory. Defaults to CHROMA_PERSIST_DIR env var.

    Returns:
        Dict with keys:
          - documents_loaded: int
          - chunks_created: int
          - persist_dir: str
          - collection_name: str
    """
    logger.info("Starting ingestion pipeline...")

    # Load documents
    documents = load_nhs_documents(docs_dir)
    num_documents = len(documents)

    # Chunk documents
    chunks = chunk_documents(documents)
    num_chunks = len(chunks)

    # Create vector store
    create_vector_store(chunks, persist_dir)

    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    logger.info("Ingestion pipeline complete!")

    return {
        "documents_loaded": num_documents,
        "chunks_created": num_chunks,
        "persist_dir": persist_dir,
        "collection_name": "nhs_guidelines",
    }
