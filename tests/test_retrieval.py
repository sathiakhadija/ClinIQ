"""
Tests for the retrieval module.
"""

from langchain_core.documents import Document

from src.retrieval import format_retrieved_chunks


class TestFormatRetrievedChunks:
    """Tests for format_retrieved_chunks function."""

    def test_format_retrieved_chunks_empty(self):
        """Test formatting empty chunk list."""
        result = format_retrieved_chunks([])

        assert isinstance(result, str)
        assert len(result) == 0 or result == ""

    def test_format_retrieved_chunks_with_mock(self):
        """Test formatting with a mock document."""
        # Create a mock Document with metadata
        mock_doc = Document(
            page_content="NHS guidelines recommend regular blood pressure monitoring.",
            metadata={
                "source": "hypertension_guidelines.pdf",
                "page": 3
            }
        )

        result = format_retrieved_chunks([mock_doc])

        assert isinstance(result, str)
        assert "hypertension_guidelines.pdf" in result
        assert "Page: 3" in result
        assert "NHS guidelines recommend" in result

    def test_format_retrieved_chunks_multiple(self):
        """Test formatting with multiple documents."""
        mock_doc1 = Document(
            page_content="First-line treatment is metformin.",
            metadata={"source": "diabetes.pdf", "page": 5}
        )
        mock_doc2 = Document(
            page_content="If metformin fails, add sulfonylurea.",
            metadata={"source": "diabetes.pdf", "page": 8}
        )

        result = format_retrieved_chunks([mock_doc1, mock_doc2])

        assert "diabetes.pdf" in result
        assert "Page: 5" in result
        assert "Page: 8" in result
        assert "metformin" in result
        assert "sulfonylurea" in result
        assert "---" in result  # Separator between chunks
