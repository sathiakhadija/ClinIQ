"""
Tests for the ingestion module.
"""

import pytest
from langchain_core.documents import Document

from src.ingestion import (
    chunk_documents,
    load_nhs_documents,
)


class TestChunkDocuments:
    """Tests for chunk_documents function."""

    def test_chunk_documents_basic(self):
        """Test basic chunking of a mock document."""
        # Create a mock LangChain Document
        mock_doc = Document(
            page_content="This is a test document about NHS guidelines for diabetes management. "
                        "Diabetes is a chronic condition. "
                        "Treatment options include insulin and oral medications. "
                        "Patients should monitor blood glucose regularly.",
            metadata={"source": "test.pdf", "page": 0}
        )

        # Chunk with small sizes for testing
        chunks = chunk_documents([mock_doc], chunk_size=50, chunk_overlap=10)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1

        # All chunks should have page_content attribute
        for chunk in chunks:
            assert hasattr(chunk, "page_content")
            assert isinstance(chunk.page_content, str)

    def test_load_nhs_documents_missing_dir(self):
        """Test that FileNotFoundError is raised for missing directory."""
        with pytest.raises(FileNotFoundError):
            load_nhs_documents("/nonexistent/path/12345")

    def test_load_nhs_documents_empty_dir(self, tmp_path):
        """Test that ValueError is raised for directory with no PDFs."""
        # Create a temporary empty directory
        docs_dir = tmp_path / "empty_docs"
        docs_dir.mkdir()

        with pytest.raises(ValueError):
            load_nhs_documents(str(docs_dir))
