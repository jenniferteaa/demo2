import itertools
import logging
from typing import Any, Dict, List, Optional

import tqdm
import torch
import numpy as np
from sentence_transformers import CrossEncoder

from mistralai import Mistral
# Initialize Mistral API
api_key = "Replace with your valid API key"  # 
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvidenceSelector:
    """
    A class to handle evidence selection using a cross-encoder for relevance scoring.
    Capable of both atomic (statement-wise) and non-atomic evidence selection.
    """

    def __init__(self):
        """
        Initialize the EvidenceSelector with a CrossEncoder passage ranker.
        We default to the 'cross-encoder/ms-marco-MiniLM-L-6-v2' model,
        running on GPU if available, else CPU.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.passage_ranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
                device=self.device
            )
            logger.info(f"Initialized CrossEncoder on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder: {e}")
            self.passage_ranker = None

    def compute_score_matrix(
        self,
        questions: List[str],
        evidences: List[str],
        context: Optional[str] = None,
        statement: Optional[str] = None,
        batch_size: int = 16,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Compute a score matrix measuring how each (question + optional context/statement) pair
        aligns with each evidence text. Returns a (len(questions) x len(evidences)) matrix of scores.

        Steps:
            1) Build a "contextualized" question that includes the context, statement, and question.
            2) Form all pairs (question_i, evidence_j) for scoring.
            3) Use the CrossEncoder to predict a relevance score for each pair.
            4) Reshape into a matrix, then normalize row-wise to [0, 1].
        """
        if not self.passage_ranker:
            logger.error("Passage ranker not initialized; returning empty matrix.")
            return np.array([])

        if not questions or not evidences:
            logger.warning("Empty questions or evidences; returning empty matrix.")
            return np.array([])

        try:
            # Build contextualized questions
            contextualized_questions = []
            for q in questions:
                parts = []
                if context:
                    parts.append(f"Context: {context}")
                if statement:
                    parts.append(f"Statement: {statement}")
                parts.append(f"Question: {q}")
                # Join them for the final input to the cross-encoder
                contextualized_questions.append(" ".join(parts))

            # Prepare (question, evidence) pairs
            pairs = [(q, e) for q in contextualized_questions for e in evidences]

            # Break into batches
            batches = [
                pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)
            ]

            all_scores = []
            # Optionally display progress with tqdm
            batch_iterator = tqdm.tqdm(batches, desc="Computing scores") if show_progress else batches

            # Predict scores for each batch
            for batch in batch_iterator:
                with torch.no_grad():
                    scores = self.passage_ranker.predict(batch)
                    all_scores.extend(scores)

            # Reshape scores into a matrix: (#questions, #evidences)
            score_matrix = np.array(all_scores).reshape(len(questions), len(evidences))

            # Normalize row-wise to [0, 1]
            return self._normalize_scores(score_matrix)

        except Exception as e:
            logger.error(f"Error computing scores: {e}")
            return np.array([])

    def _normalize_scores(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalizes each row to the [0, 1] range.
        If row min == max, we set that entire row to 0.5 (if max>0) or 0.0 (otherwise).
        """
        try:
            normalized = np.zeros_like(matrix)
            for i, row in enumerate(matrix):
                min_val, max_val = row.min(), row.max()
                if max_val > min_val:
                    normalized[i] = (row - min_val) / (max_val - min_val)
                else:
                    # Fallback for uniform or zero row
                    normalized[i] = np.ones_like(row) * (0.5 if max_val > 0 else 0.0)
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing scores: {e}")
            return matrix  # Return unnormalized matrix if something fails

    def _filter_irrelevant_evidence(
        self,
        evidences: List[Dict[str, Any]],
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Filters out evidence items whose 'score' is below `min_score`.
        """
        return [e for e in evidences if e.get('score', 0) >= min_score]

    def select_evidences_for_atomic_statement(
        self,
        questions: List[str],
        evidences: List[str],
        context: Optional[str] = None,
        statement: Optional[str] = None,
        max_selected: int = 5,
        min_score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Select evidences relevant to a single atomic statement by:
            1) Generating a score matrix (#questions x #evidences).
            2) Taking the max score for each evidence across all questions.
            3) Filtering out evidences below `min_score_threshold`.
            4) Sorting by descending score and returning top `max_selected`.

        Returns a list of dicts with 'text', 'score', and optionally 'relevant_questions'.
        """
        try:
            if not questions or not evidences:
                return []

            # Compute the (questions x evidences) relevance matrix
            score_matrix = self.compute_score_matrix(
                questions,
                evidences,
                context=context,
                statement=statement
            )
            if score_matrix.size == 0:
                return []

            # For each evidence (column), take the max score across all questions (rows)
            max_scores = np.max(score_matrix, axis=0)

            evidence_info = []
            for i, evidence_text in enumerate(evidences):
                if max_scores[i] >= min_score_threshold:
                    # Gather question-level relevance for any question above threshold
                    relevant_qs = [
                        {
                            'question': questions[q],
                            'relevance_score': float(score_matrix[q, i])
                        }
                        for q in np.where(score_matrix[:, i] >= min_score_threshold)[0]
                    ]
                    if relevant_qs:
                        evidence_info.append({
                            'text': evidence_text,
                            'score': float(max_scores[i]),
                            'relevant_questions': relevant_qs
                        })

            # Sort by descending overall score
            evidence_info.sort(key=lambda x: x['score'], reverse=True)

            # Return the top `max_selected`
            return evidence_info[:max_selected]

        except Exception as e:
            logger.error(f"Error in select_evidences_for_atomic_statement: {e}")
            return []

    def process_atomic_statements(
        self,
        statements: List[str],
        questions_per_statement: List[List[str]],
        evidences: List[str],
        max_evidences_per_statement: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Iterate over each statement, fetch relevant evidences for it, and maintain
        a running context of previously processed statements.

        For each statement i:
            1) Build a context string from all previously processed statements (0..i-1).
            2) Call `select_evidences_for_atomic_statement`.
            3) Append the result, update the context to include this statement.

        Returns a list of dicts with:
            {
                "statement": str,
                "questions": [...],
                "evidences": [...],
                "context_used": bool,
            }
        """
        results = []
        context = ""

        for i, (statement, questions) in enumerate(zip(statements, questions_per_statement)):
            logger.info(f"Processing statement {i + 1}/{len(statements)}")

            # For statement i, build context from all previously processed statements
            context_for_this = context if i > 0 else None

            selected_evidences = self.select_evidences_for_atomic_statement(
                questions=questions,
                evidences=evidences,
                context=context_for_this,
                statement=statement,
                max_selected=max_evidences_per_statement
            )
            
            results.append({
                "statement": statement,
                "questions": questions,
                "evidences": selected_evidences,
                "context_used": bool(context_for_this)
            })
            
            # Update the context to include the current statement as well
            # (Some pipelines may prefer using the revised statement instead, if available)
            context = " ".join(statements[:i + 1])

        return results

    def select_evidences(
        self,
        example: Dict[str, Any],
        max_selected: int = 5,
        atomic_processing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Main public method to select evidences from an `example`. If `atomic_processing` is True
        and `example` contains `atomic_statements` plus `questions_per_statement`, we process them
        statement-by-statement. Otherwise, a simpler (non-atomic) approach is used.

        Non-atomic approach:
            - Combine all questions
            - Score each evidence
            - Return top `max_selected` by relevance.
        """
        try:
            # Extract the possible evidence texts from 'revisions'
            evidences = self._extract_evidences(example)
            if not evidences:
                logger.warning("No evidences found in example.")
                return []

            if atomic_processing and "atomic_statements" in example:
                statements = example["atomic_statements"]
                questions_per_statement = example.get("questions_per_statement", [])
                
                # Validate lengths
                if len(statements) != len(questions_per_statement):
                    logger.warning(
                        "Mismatched lengths for statements and questions_per_statement. "
                        "Results may be incomplete."
                    )

                return self.process_atomic_statements(
                    statements=statements,
                    questions_per_statement=questions_per_statement,
                    evidences=evidences,
                    max_evidences_per_statement=max_selected
                )
            else:
                # Non-atomic fallback
                logger.info("Using non-atomic processing mode.")
                questions = sorted(set(example.get("questions", [])))
                if not questions:
                    logger.warning("No questions found for non-atomic processing.")
                    return []

                # Compute scores in a single pass
                score_matrix = self.compute_score_matrix(questions, evidences)
                if score_matrix.size == 0:
                    return []

                # For each evidence, take the max score across all questions, then pick top `max_selected`
                max_scores = np.max(score_matrix, axis=0)
                best_indices = np.argsort(-max_scores)[:max_selected]

                return [
                    {
                        "text": evidences[i],
                        "score": float(max_scores[i])
                    }
                    for i in best_indices
                ]

        except Exception as e:
            logger.error(f"Error in select_evidences: {e}")
            return []

    def _extract_evidences(self, example: Dict[str, Any]) -> List[str]:
        """
        Helper to retrieve evidence strings from the 'revisions' field of an example.
        Ensures each piece of evidence is non-empty and of sufficient length.
        """
        try:
            revisions = example.get("revisions", [])
            if not revisions or "evidences" not in revisions[0]:
                return []
            
            # Gather all distinct evidence texts
            evidences = []
            for e in revisions[0]["evidences"]:
                if isinstance(e, dict) and e.get("text"):
                    text = e["text"].strip()
                    # A basic check to exclude extremely short or invalid entries
                    if text and len(text) > 10:
                        evidences.append(text)
            
            # Return unique sorted evidences
            return sorted(set(evidences))

        except Exception as e:
            logger.error(f"Error extracting evidences: {e}")
            return []


if __name__ == "__main__":
    # Simple usage example with Lena Headey claims
    selector = EvidenceSelector()

    sample_statements = [
        "Lena Headey portrayed Cersei Lannister in HBO's Game of Thrones since 2011.",
        "She received three consecutive Emmy nominations for Outstanding Supporting Actress.",
        "By 2017, she became one of the highest-paid television actors."
    ]

    sample_questions = [
        [
            "When did Lena Headey start playing Cersei Lannister?",
            "Was she the original actress cast for the role?",
            "How long did she play the character?"
        ],
        [
            "Which years did she receive Emmy nominations?",
            "Were the nominations consecutive?",
            "What was the exact category for her nominations?"
        ],
        [
            "What was her reported salary by 2017?",
            "How did her salary compare to other TV actors?",
            "What factors contributed to her salary increase?"
        ]
    ]

    # Example usage of evidence selection
    test_example = {
        "atomic_statements": sample_statements,
        "questions_per_statement": sample_questions,
        "revisions": [{
            "evidences": [{"text": "Sample evidence " + str(i)} for i in range(5)]
        }]
    }

    results = selector.select_evidences(test_example, atomic_processing=True)
    
    print("\nResults per Statement:")
    for result in results:
        print(f"\nStatement: {result['statement']}")
        print("Selected Evidences:")
        for evidence in result.get('evidences', []):
            print(f"- Evidence: {evidence['text']}")
            score_str = f"{evidence.get('score', 0.0):.3f}"
            print(f"  Score: {score_str}")
            if 'relevant_questions' in evidence:
                print("  Relevant Questions:")
                for q_info in evidence['relevant_questions']:
                    print(f"    * {q_info['question']} (score: {q_info['relevance_score']:.3f})")

