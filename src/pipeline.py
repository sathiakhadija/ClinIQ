"""
End-to-end RAG pipeline for ClinIQ.

Combines ingestion, retrieval, and generation into a single interface.
Optionally logs to Weights & Biases Weave.
"""

import logging
from dotenv import load_dotenv

from src.retrieval import get_retriever, retrieve_chunks, format_retrieved_chunks
from src.generation import generate_answer, format_response

try:
    import weave
except ImportError:
    weave = None

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class ClinIQPipeline:
    """
    End-to-end RAG pipeline for ClinIQ.

    Combines retrieval and generation into a single callable interface.
    Optionally logs queries and responses to Weights & Biases Weave.
    """

    def __init__(self, vector_store, use_weave: bool = True):
        """
        Initialise with an existing vector store.

        Args:
            vector_store: ChromaDB Chroma instance (from ingestion.load_existing_vector_store()).
            use_weave: If True, initialise W&B Weave logging.

        Sets:
            self.vector_store: The vector store
            self.retriever: LangChain retriever from the vector store
            self.use_weave: Whether to log to Weave
        """
        self.vector_store = vector_store
        self.retriever = get_retriever(vector_store)
        self.use_weave = use_weave

        if use_weave and weave is not None:
            # Initialize W&B Weave
            try:
                weave.init("cliniq")
                logger.info("W&B Weave initialised for logging")
            except Exception as e:
                logger.warning(f"Could not initialise W&B Weave: {e}")
                self.use_weave = False
        elif use_weave:
            logger.warning("W&B Weave is not installed; logging disabled")
            self.use_weave = False

    def query(self, question: str) -> dict:
        """
        Run the full RAG pipeline for a single question.

        Steps:
          1. Retrieve relevant chunks
          2. Format context
          3. Generate answer
          4. Format response
          5. Log to Weave if enabled

        Args:
            question: User question string.

        Returns:
            Dict with keys:
              - answer: Generated answer
              - sources: List of source dicts with filename and page
        """
        logger.info(f"Processing query: {question}")

        # Step 1: Retrieve relevant chunks
        chunks = retrieve_chunks(question, self.retriever)

        # Step 2: Format context
        context = format_retrieved_chunks(chunks)

        # Step 3: Generate answer
        generation_result = generate_answer(question, context)
        answer = generation_result["answer"]
        token_info = {
            "prompt_tokens": generation_result["prompt_tokens"],
            "completion_tokens": generation_result["completion_tokens"],
            "total_tokens": generation_result["total_tokens"],
        }

        # Step 4: Format response
        response = format_response(answer, chunks)

        # Step 5: Log to Weave if enabled
        if self.use_weave:
            try:
                weave.log({
                    "question": question,
                    "answer": answer,
                    "sources": response["sources"],
                    "tokens_used": token_info,
                    "num_chunks_retrieved": len(chunks),
                })
                logger.info("Logged to W&B Weave")
            except Exception as e:
                logger.warning(f"Could not log to W&B Weave: {e}")

        return response

    def batch_query(self, questions: list) -> list:
        """
        Run the pipeline for a list of questions.

        Args:
            questions: List of question strings.

        Returns:
            List of response dicts (one per question).

        Logs:
            - Progress every 5 questions
        """
        logger.info(f"Processing batch of {len(questions)} questions")

        responses = []
        for i, question in enumerate(questions):
            response = self.query(question)
            responses.append(response)

            # Log progress
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(questions)} questions")

        logger.info(f"Batch processing complete. {len(responses)} responses generated.")
        return responses
