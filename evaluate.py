"""
CLI script to run RAGAS evaluation on the ClinIQ pipeline.

Evaluates retrieval and generation quality using RAGAS metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Usage:
  python evaluate.py
  python evaluate.py --output ./my_results.json

Requires the vector store to be populated first (run ingest.py).
"""

import sys
import argparse
import logging

from src.ingestion import load_existing_vector_store
from src.pipeline import ClinIQPipeline
from src.evaluation import build_evaluation_dataset, run_ragas_evaluation, save_evaluation_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_results_table(results: dict):
    """Print evaluation results as a formatted table."""
    print("\n" + "=" * 70)
    print("RAGAS Evaluation Results")
    print("=" * 70)
    print(f"{'Metric':<25} {'Score':<10} {'Interpretation':<35}")
    print("-" * 70)

    interpretations = {
        "faithfulness": _interpret_score(results.get("faithfulness", 0), "faithfulness"),
        "answer_relevancy": _interpret_score(results.get("answer_relevancy", 0), "answer_relevancy"),
        "context_precision": _interpret_score(results.get("context_precision", 0), "context_precision"),
        "context_recall": _interpret_score(results.get("context_recall", 0), "context_recall"),
    }

    for metric, score in results.items():
        interpretation = interpretations[metric]
        print(f"{metric:<25} {score:.3f}      {interpretation:<35}")

    print("=" * 70)


def _interpret_score(score: float, metric: str) -> str:
    """Return a brief interpretation of the score."""
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.8:
        return "Good"
    elif score >= 0.7:
        return "Fair"
    else:
        return "Poor - Improve"


def main():
    """Main entry point for evaluation CLI."""

    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the ClinIQ pipeline"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_results.json",
        help="Path to save evaluation results (default: ./evaluation_results.json)"
    )

    args = parser.parse_args()

    try:
        logger.info("Starting RAGAS evaluation...")

        # Load existing vector store
        logger.info("Loading existing vector store...")
        vector_store = load_existing_vector_store()

        # Initialize pipeline
        logger.info("Initializing ClinIQ pipeline...")
        pipeline = ClinIQPipeline(vector_store, use_weave=True)

        # Build evaluation dataset
        logger.info("Building evaluation dataset...")
        dataset = build_evaluation_dataset(pipeline)

        # Run RAGAS evaluation
        logger.info("Running RAGAS evaluation...")
        results = run_ragas_evaluation(dataset)

        # Save results
        logger.info(f"Saving results to {args.output}...")
        save_evaluation_results(results, args.output)

        # Print results table
        print_results_table(results)

        logger.info(f"Evaluation complete. Results saved to {args.output}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Vector store not found. Please run ingest.py first to populate the vector store.")
        logger.error("Usage: python ingest.py --docs-dir ./data/nhs_docs")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
