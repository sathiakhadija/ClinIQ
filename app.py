"""
Streamlit chat interface for ClinIQ.

A web-based chat application for querying NHS clinical guidelines.
Loads the vector store on startup and processes queries in real-time.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if __name__ == "__main__":
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
        logging.ERROR
    )

if __name__ == "__main__" and get_script_run_ctx() is None:
    print("ClinIQ is a Streamlit app. Start it with: streamlit run app.py")
    sys.exit(1)

from src.ingestion import load_existing_vector_store
from src.pipeline import ClinIQPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ClinIQ — NHS Guidelines Intelligence",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-left: 4px solid #0072CE;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    if "vector_store_loaded" not in st.session_state:
        st.session_state.vector_store_loaded = False


def load_vector_store_and_pipeline():
    """Load the vector store and initialize the pipeline."""
    try:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

        if not Path(persist_dir).exists():
            st.error(
                f"❌ Vector store not found at {persist_dir}\n\n"
                "To use ClinIQ, you must first:\n"
                "1. Download NHS PDF guidelines from https://www.nice.org.uk/guidance\n"
                "2. Place them in the data/nhs_docs/ directory\n"
                "3. Run: `python ingest.py`\n\n"
                "This will create the vector store. Then refresh this page."
            )
            return False

        logger.info("Loading vector store...")
        vector_store = load_existing_vector_store(persist_dir)

        logger.info("Initializing pipeline...")
        st.session_state.pipeline = ClinIQPipeline(vector_store, use_weave=True)
        st.session_state.vector_store_loaded = True

        logger.info("Vector store and pipeline ready")
        return True

    except FileNotFoundError:
        st.error(
            "❌ Vector store not found.\n\n"
            "Please run `python ingest.py` to populate the vector store."
        )
        return False

    except Exception as e:
        st.error(f"❌ Error loading vector store: {e}")
        logger.error(f"Error loading vector store: {e}", exc_info=True)
        return False


def render_sidebar():
    """Render the sidebar with instructions, examples, and settings."""
    with st.sidebar:
        # Logo and title
        st.markdown("## 🏥 ClinIQ")
        st.markdown("**NHS Guidelines Intelligence**")
        st.markdown("---")

        # How it works
        st.markdown("### How it works")
        st.markdown("""
        1. **Ask a question** about NHS clinical guidelines
        2. **ClinIQ retrieves** relevant guideline sections
        3. **GPT-4o-mini generates** a sourced answer
        """)

        st.markdown("---")

        # Example questions
        st.markdown("### Example questions")
        example_questions = [
            "What are the NICE guidelines for hypertension?",
            "How should type 2 diabetes be managed?",
            "What is the recommended treatment for depression?",
            "When should statins be prescribed?",
        ]

        for i, question in enumerate(example_questions):
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.selected_question = question

        st.markdown("---")

        # Clear conversation button
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        # Disclaimer
        st.markdown("### ⚠️ Disclaimer")
        st.markdown(
            "ClinIQ provides information from NHS guidelines only. "
            "Always consult a qualified healthcare professional for medical advice."
        )


def display_messages():
    """Display chat messages from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if this is an assistant message
            if message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.markdown(
                                f"• **{source['source']}** (Page {source['page']})"
                            )


def process_user_query(user_input: str):
    """Process a user query and generate a response."""
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching NHS guidelines..."):
            try:
                response = st.session_state.pipeline.query(user_input)

                answer = response["answer"]
                sources = response["sources"]

                st.markdown(answer)

                # Display sources
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.markdown(
                                f"• **{source['source']}** (Page {source['page']})"
                            )

                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_message = f"❌ Error generating response: {e}"
                st.error(error_message)
                logger.error(f"Error generating response: {e}", exc_info=True)

                # Add error to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })


def main():
    """Main application entry point."""

    # Initialize session state
    initialize_session_state()

    # Render sidebar
    render_sidebar()

    # Main content area
    st.markdown("# ClinIQ")
    st.markdown("**Intelligent NHS Clinical Guidelines Assistant**")
    st.markdown("---")

    # Load vector store on first run
    if not st.session_state.vector_store_loaded:
        load_vector_store_and_pipeline()

    # Check if vector store is loaded
    if not st.session_state.vector_store_loaded:
        st.stop()

    # Display existing messages
    display_messages()

    # Check if an example question was selected
    if "selected_question" in st.session_state:
        user_input = st.session_state.selected_question
        del st.session_state.selected_question
        process_user_query(user_input)

    # Chat input at bottom
    user_input = st.chat_input("Ask me about NHS clinical guidelines...")

    if user_input:
        process_user_query(user_input)


if __name__ == "__main__":
    main()
