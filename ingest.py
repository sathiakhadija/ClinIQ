"""
CLI script to run the ClinIQ ingestion pipeline.

Loads NHS PDFs, chunks them, embeds them, and stores in ChromaDB.

Usage:
  python ingest.py
  python ingest.py --docs-dir ./data/nhs_docs
  python ingest.py --docs-dir ./data/nhs_docs --persist-dir ./data/chroma

Place NHS PDF files in the docs directory before running.
Download NHS guidelines from: https://www.nice.org.uk/guidance
"""

import sys
import argparse
import logging

from src.ingestion import run_ingestion_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for ingestion CLI."""

    parser = argparse.ArgumentParser(
        description="Run the ClinIQ ingestion pipeline"
    )

    parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="Path to NHS documents directory (default: env var NHS_DOCS_DIR)"
    )

    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="Path to ChromaDB persistence directory (default: env var CHROMA_PERSIST_DIR)"
    )

    args = parser.parse_args()

    try:
        logger.info("Starting ingestion pipeline...")

        result = run_ingestion_pipeline(
            docs_dir=args.docs_dir,
            persist_dir=args.persist_dir
        )

        logger.info("\n" + "=" * 60)
        logger.info("✓ Ingestion complete!")
        logger.info("=" * 60)
        logger.info(f"Documents loaded: {result['documents_loaded']}")
        logger.info(f"Chunks created: {result['chunks_created']}")
        logger.info(f"Persist directory: {result['persist_dir']}")
        logger.info(f"Collection name: {result['collection_name']}")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please ensure the NHS documents directory exists and contains PDF files.")
        logger.error("Download from: https://www.nice.org.uk/guidance")
        return 1

    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error("No PDF files found in the specified directory.")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
