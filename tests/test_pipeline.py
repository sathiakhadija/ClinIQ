"""
Tests for the generation and pipeline modules.
"""

from src.generation import build_prompt, format_response
from langchain_core.documents import Document


class TestGenerationModule:
    """Tests for the generation module."""

    def test_build_prompt_structure(self):
        """Test that build_prompt returns correct structure."""
        query = "What is the treatment for diabetes?"
        context = "Context: NHS recommends metformin."

        messages = build_prompt(query, context)

        assert isinstance(messages, list)
        assert len(messages) >= 2

        # Check for system role
        assert any(msg["role"] == "system" for msg in messages)

        # Check for user role
        assert any(msg["role"] == "user" for msg in messages)

        # Check that query is in user message
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        assert any(query in msg["content"] for msg in user_messages)

    def test_build_prompt_includes_context(self):
        """Test that context is included in the prompt."""
        query = "What is diabetes?"
        context = "NHS defines diabetes as..."

        messages = build_prompt(query, context)

        # Context should be in one of the messages
        all_content = " ".join([msg["content"] for msg in messages])
        assert context in all_content

    def test_format_response_deduplication(self):
        """Test that format_response deduplicates sources."""
        # Create chunks with duplicate sources
        chunk1 = Document(
            page_content="First information",
            metadata={"source": "guidelines.pdf", "page": 5}
        )
        chunk2 = Document(
            page_content="Second information",
            metadata={"source": "guidelines.pdf", "page": 5}
        )
        chunk3 = Document(
            page_content="Third information",
            metadata={"source": "guidelines.pdf", "page": 8}
        )

        answer = "Test answer"
        response = format_response(answer, [chunk1, chunk2, chunk3])

        assert response["answer"] == answer
        assert isinstance(response["sources"], list)

        # Should have 2 unique sources (page 5 and page 8)
        assert len(response["sources"]) == 2

    def test_format_response_structure(self):
        """Test that format_response returns correct structure."""
        chunk = Document(
            page_content="Some medical information",
            metadata={"source": "medical_guidelines.pdf", "page": 10}
        )

        answer = "This is the answer"
        response = format_response(answer, [chunk])

        assert "answer" in response
        assert "sources" in response
        assert response["answer"] == answer
        assert isinstance(response["sources"], list)

        if response["sources"]:
            source = response["sources"][0]
            assert "source" in source
            assert "page" in source
