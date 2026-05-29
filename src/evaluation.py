"""
RAGAS evaluation for ClinIQ.

Evaluates the RAG pipeline using RAGAS metrics:
- Faithfulness: Is answer grounded in context?
- Answer Relevancy: Does answer address the question?
- Context Precision: Are retrieved chunks relevant?
- Context Recall: Does context cover the needed information?
"""

import json
import logging
from datetime import datetime
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Evaluation questions to assess ClinIQ quality
EVALUATION_QUESTIONS = [
    "What is the recommended first-line treatment for type 2 diabetes?",
    "What are the NICE guidelines for hypertension management?",
    "When should antibiotics be prescribed for respiratory infections?",
    "What are the diagnostic criteria for depression according to NHS guidelines?",
    "What is the recommended screening frequency for cervical cancer?",
    "How should acute asthma be managed in adults?",
    "What are the NICE recommendations for statin therapy?",
    "When is referral to secondary care recommended for chest pain?",
    "What lifestyle interventions are recommended for obesity management?",
    "What are the NHS guidelines for managing chronic kidney disease?"
]


def build_evaluation_dataset(pipeline, questions: list = None) -> dict:
    """
    Run the pipeline on evaluation questions to build the RAGAS dataset.

    Args:
        pipeline: ClinIQPipeline instance.
        questions: List of questions. Defaults to EVALUATION_QUESTIONS.

    Returns:
        Dict with keys:
          - questions: List of question strings
          - answers: List of generated answer strings
          - contexts: List of lists of chunk strings (retrieved chunks per question)
          - ground_truths: List of placeholder ground truth strings
    """
    if questions is None:
        questions = EVALUATION_QUESTIONS

    logger.info(f"Building evaluation dataset with {len(questions)} questions...")

    answers = []
    contexts = []
    ground_truths = []

    for i, question in enumerate(questions):
        logger.info(f"  Processing question {i+1}/{len(questions)}: {question[:50]}...")

        # Run pipeline
        response = pipeline.query(question)
        answer = response["answer"]
        answers.append(answer)

        # Extract context chunks (retrieve from pipeline)
        # We need to run retrieval again to get the chunks
        from src.retrieval import retrieve_chunks
        chunks = retrieve_chunks(question, pipeline.retriever)
        context_strings = [chunk.page_content for chunk in chunks]
        contexts.append(context_strings)

        # Ground truth is placeholder (see Task 7 discussion)
        ground_truths.append("See NHS guidelines")

    logger.info(f"Built dataset: {len(questions)} questions, {len(answers)} answers")

    return {
        "questions": questions,
        "answers": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
    }


def run_ragas_evaluation(dataset: dict) -> dict:
    """
    Run RAGAS evaluation on the dataset.

    Evaluates:
      - Faithfulness: Is answer grounded in context?
      - Answer Relevancy: Does answer address the question?
      - Context Precision: Are retrieved chunks relevant?
      - Context Recall: Does context cover the answer?

    Args:
        dataset: Dict from build_evaluation_dataset() with keys:
                 questions, answers, contexts, ground_truths

    Returns:
        Dict with keys:
          - faithfulness: Score (0-1)
          - answer_relevancy: Score (0-1)
          - context_precision: Score (0-1)
          - context_recall: Score (0-1)

    Logs:
        - Scores to logger
        - Scores to W&B Weave
    """
    logger.info("Starting RAGAS evaluation...")

    # Convert to Hugging Face Dataset format (required by RAGAS)
    hf_dataset = Dataset.from_dict({
        "question": dataset["questions"],
        "answer": dataset["answers"],
        "contexts": dataset["contexts"],
        "ground_truth": dataset["ground_truths"],
    })

    # Run evaluation
    logger.info("Evaluating faithfulness...")
    logger.info("Evaluating answer_relevancy...")
    logger.info("Evaluating context_precision...")
    logger.info("Evaluating context_recall...")

    result = evaluate(
        hf_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    )

    # Extract scores
    scores = {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
        "context_recall": result["context_recall"],
    }

    # Log scores
    logger.info("Evaluation complete:")
    logger.info(f"  Faithfulness: {scores['faithfulness']:.3f}")
    logger.info(f"  Answer Relevancy: {scores['answer_relevancy']:.3f}")
    logger.info(f"  Context Precision: {scores['context_precision']:.3f}")
    logger.info(f"  Context Recall: {scores['context_recall']:.3f}")

    # Log to W&B Weave if available
    try:
        import weave
        weave.log({
            "evaluation_metrics": scores,
            "num_questions": len(dataset["questions"]),
        })
        logger.info("Logged evaluation results to W&B Weave")
    except Exception as e:
        logger.warning(f"Could not log to W&B Weave: {e}")

    return scores


def save_evaluation_results(results: dict, output_path: str = None):
    """
    Save evaluation results to a JSON file.

    Args:
        results: Dict from run_ragas_evaluation() with metric scores.
        output_path: Path to save JSON. Defaults to ./evaluation_results.json

    Includes:
        - Timestamp
        - Metric scores
        - Brief interpretation of each score
    """
    if output_path is None:
        output_path = "./evaluation_results.json"

    # Build output dict with interpretation
    output = {
        "timestamp": datetime.now().isoformat(),
        "metrics": results,
        "interpretation": {
            "faithfulness": _interpret_faithfulness(results.get("faithfulness", 0)),
            "answer_relevancy": _interpret_answer_relevancy(results.get("answer_relevancy", 0)),
            "context_precision": _interpret_context_precision(results.get("context_precision", 0)),
            "context_recall": _interpret_context_recall(results.get("context_recall", 0)),
        }
    }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Evaluation results saved to {output_path}")


def _interpret_faithfulness(score: float) -> str:
    """Interpret faithfulness score (0-1)."""
    if score >= 0.9:
        return "Excellent: Answers are highly grounded in retrieved context."
    elif score >= 0.8:
        return "Good: Answers are mostly grounded in context."
    elif score >= 0.7:
        return "Fair: Some answers contain unsupported claims."
    else:
        return "Poor: Many answers lack grounding in context. Investigate retrieval quality."


def _interpret_answer_relevancy(score: float) -> str:
    """Interpret answer relevancy score (0-1)."""
    if score >= 0.9:
        return "Excellent: Answers directly address questions asked."
    elif score >= 0.8:
        return "Good: Answers mostly address the question."
    elif score >= 0.7:
        return "Fair: Some answers are tangential or incomplete."
    else:
        return "Poor: Answers often miss the question. Improve prompt engineering."


def _interpret_context_precision(score: float) -> str:
    """Interpret context precision score (0-1)."""
    if score >= 0.9:
        return "Excellent: Retrieved chunks are highly relevant."
    elif score >= 0.8:
        return "Good: Most retrieved chunks are relevant."
    elif score >= 0.7:
        return "Fair: Some retrieved chunks are noisy. Consider increasing top_k filtering."
    else:
        return "Poor: Retrieved chunks often irrelevant. Check embedding quality or increase similarity threshold."


def _interpret_context_recall(score: float) -> str:
    """Interpret context recall score (0-1)."""
    if score >= 0.9:
        return "Excellent: Retrieved context covers what's needed to answer."
    elif score >= 0.8:
        return "Good: Retrieved context mostly sufficient."
    elif score >= 0.7:
        return "Fair: Some questions need more context. Consider increasing top_k."
    else:
        return "Poor: Retrieved context often insufficient. Increase top_k or improve chunking strategy."
