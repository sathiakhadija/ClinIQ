"""
Generation module for ClinIQ.

Builds prompts and generates answers using GPT-4o-mini.
"""

import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

def get_openai_client() -> OpenAI:
    """Create the OpenAI client lazily so tests can import without real secrets."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your OpenAI API key."
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt for the LLM
SYSTEM_PROMPT = """You are a clinical information assistant specialising in NHS guidelines. Your role is to help healthcare professionals and patients understand NHS clinical guidelines accurately.

Rules:
- Only answer based on the provided NHS guideline context
- Always cite the specific guideline and page number for every claim
- If the context does not contain enough information to answer, say so clearly
- Never provide medical advice beyond what is stated in the guidelines
- Use clear, accessible language appropriate for both clinicians and patients
- If the question is outside the scope of NHS guidelines, say so

Format your response as:
1. A direct answer to the question
2. Supporting details from the guidelines
3. Citations: [Guideline name, Page X]"""


def build_prompt(query: str, context: str) -> list:
    """
    Build the message list for the OpenAI chat completion.

    Args:
        query: User question string.
        context: Formatted context from retrieve_chunks().

    Returns:
        List of message dicts with role and content.
        Includes system prompt, context, and user query.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context from NHS guidelines:\n\n{context}\n\nQuestion: {query}"
        }
    ]

    return messages


def generate_answer(query: str, context: str, model: str = None) -> dict:
    """
    Call the OpenAI API to generate an answer.

    Args:
        query: User question string.
        context: Formatted context from retrieve_chunks().
        model: Model name. Defaults to MODEL_NAME env var (gpt-4o-mini).

    Returns:
        Dict with keys:
          - answer: Generated answer string
          - model: Model used
          - prompt_tokens: Input token count
          - completion_tokens: Output token count
          - total_tokens: Total token count

    Logs:
        - Token usage after each call
    """
    if model is None:
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    messages = build_prompt(query, context)

    # Call OpenAI API
    response = get_openai_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
    )

    # Extract answer and token usage
    answer = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    logger.info(
        f"Generated answer using {model}. "
        f"Tokens: {prompt_tokens} input + {completion_tokens} output = {total_tokens} total"
    )

    return {
        "answer": answer,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def format_response(answer: str, chunks: list) -> dict:
    """
    Format the final response for display in the Streamlit UI.

    Args:
        answer: Generated answer from generate_answer().
        chunks: List of Document objects from retrieve_chunks().

    Returns:
        Dict with keys:
          - answer: Generated answer string
          - sources: List of dicts, each with filename and page
                    (deduplicated)
    """
    # Extract sources from chunks metadata
    sources = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown source")
        page = chunk.metadata.get("page", "Unknown page")
        key = (source, page)

        if key not in sources:
            sources[key] = {"source": source, "page": page}

    # Convert to list of dicts
    source_list = list(sources.values())

    return {
        "answer": answer,
        "sources": source_list,
    }
