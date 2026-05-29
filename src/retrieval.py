"""
Retrieval module for ClinIQ.

Retrieves relevant NHS guideline chunks for user queries.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


def get_retriever(vector_store, top_k: int = None):
    """
    Create a retriever from the vector store.

    Args:
        vector_store: ChromaDB Chroma instance.
        top_k: Number of top chunks to retrieve. Defaults to TOP_K env var (5).

    Returns:
        LangChain retriever object.
    """
    if top_k is None:
        top_k = int(os.getenv("TOP_K", "5"))

    # Create retriever with similarity search (not MMR)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    return retriever


def retrieve_chunks(query: str, retriever) -> list:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query: User question string.
        retriever: LangChain retriever object.

    Returns:
        List of Document objects with content and metadata.

    Logs:
        - The query
        - Number of chunks retrieved
    """
    logger.info(f"Retrieving chunks for query: {query}")

    chunks = retriever.invoke(query)

    logger.info(f"Retrieved {len(chunks)} chunks")

    return chunks


def format_retrieved_chunks(chunks: list) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.

    Each chunk is formatted as:
    [Source: {filename}, Page: {page}]
    {chunk content}
    ---

    Args:
        chunks: List of Document objects from retriever.

    Returns:
        Formatted context string.
    """
    if not chunks:
        return ""

    formatted_chunks = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown source")
        page = chunk.metadata.get("page", "Unknown page")
        content = chunk.page_content

        formatted_chunk = f"[Source: {source}, Page: {page}]\n{content}\n---"
        formatted_chunks.append(formatted_chunk)

    # Join all chunks with a newline separator
    context = "\n\n".join(formatted_chunks)

    return context
