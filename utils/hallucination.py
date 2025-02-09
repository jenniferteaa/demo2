import itertools
import os
import time
import logging
from typing import Any, Dict, List, Tuple, Optional

import torch
from sentence_transformers import CrossEncoder
import openai

from mistralai import Mistral
# Initialize Mistral API
api_key = "Replace with your valid API key"  # 
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize cross-encoder
PASSAGE_RANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

def compute_score_matrix(
    questions: List[str], 
    evidences: List[str],
    context: Optional[str] = None
) -> List[List[float]]:
    try:
        score_matrix = []
        for q in questions:
            contextualized_q = f"{context} {q}" if context else q
            evidence_pairs = [(contextualized_q, e) for e in evidences]

            with torch.no_grad():
                scores = PASSAGE_RANKER.predict(evidence_pairs).tolist()
            score_matrix.append(scores)
        return score_matrix
    except Exception as e:
        logger.error(f"Error computing score matrix: {e}")
        return [[0.0]*len(evidences) for _ in questions]

def question_coverage_objective_fn(
    score_matrix: List[List[float]], 
    evidence_indices: List[int],
    question_groups: Optional[List[List[int]]] = None
) -> float:
    try:
        total = 0.0
        if question_groups:
            for group in question_groups:
                group_scores = [score_matrix[q_idx] for q_idx in group]
                group_total = sum(
                    max(scores_for_question[j] for j in evidence_indices)
                    for scores_for_question in group_scores
                )
                total += group_total / len(group)
        else:
            for scores_for_question in score_matrix:
                total += max(scores_for_question[j] for j in evidence_indices)
        return total
    except Exception as e:
        logger.error(f"Error computing coverage objective: {e}")
        return 0.0

def select_evidences(
    example: Dict[str, Any], 
    max_selected: int = 5, 
    prefer_fewer: bool = False,
    atomic_statements: Optional[List[str]] = None,
    questions_per_statement: Optional[List[List[str]]] = None
) -> List[Dict[str, Any]]:
    try:
        if questions_per_statement:
            questions = [q for sublist in questions_per_statement for q in sublist]
            question_groups = []
            idx = 0
            for stmt_qs in questions_per_statement:
                group = list(range(idx, idx + len(stmt_qs)))
                question_groups.append(group)
                idx += len(stmt_qs)
        else:
            questions = sorted(set(example.get("questions", [])))
            question_groups = None

        # Gather unique evidence texts
        evidences = []
        if example.get("revisions") and example["revisions"][0].get("evidences"):
            evidences = sorted(set(e["text"] for e in example["revisions"][0]["evidences"] if e.get("text")))

        if not evidences:
            logger.warning("No evidences found in the example.")
            return []

        context = None
        if atomic_statements and len(atomic_statements) > 1:
            context = " ".join(atomic_statements[:-1])

        score_matrix = compute_score_matrix(questions, evidences, context)

        best_combo = tuple()
        best_value = float("-inf")
        max_selected = min(max_selected, len(evidences))

        min_selected = 1 if prefer_fewer else max_selected
        for num_selected in range(min_selected, max_selected+1):
            for combo in itertools.combinations(range(len(evidences)), num_selected):
                val = question_coverage_objective_fn(score_matrix, combo, question_groups)
                if val > best_value:
                    best_combo = combo
                    best_value = val

        return [
            {"text": evidences[i], "score": float(best_value)} for i in best_combo
        ]
    except Exception as e:
        logger.error(f"Error selecting evidences: {e}")
        return []

def run_evidence_hallucination(
    query: str,
    model: str,
    prompt: str,
    context: Optional[str] = None,
    atomic_statement: Optional[str] = None,
    num_retries: int = 5
) -> Dict[str, str]:
    try:
        if context and atomic_statement:
            system_content = (
                "You are a fact-checking assistant. Generate evidence based on factual "
                "information. Focus on verifiable facts and reliable sources."
            )
            full_context = (
                f"Context: {context}\n"
                f"Current statement to verify: {atomic_statement}\n"
                f"Query: {query}"
            )
            gpt_input = prompt.format(context=full_context).strip()
        else:
            system_content = "You are a helpful assistant focused on factual accuracy."
            gpt_input = prompt.format(query=query).strip()

        for attempt in range(num_retries):
            try:
                # Option A: Directly call openai
                response = client.chat.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": gpt_input},
                    ],
                    temperature=0.0,
                    max_tokens=256
                )
                #print("This is the raw response from the run_evidence_hallucination function in the hallucination.py file: ", response)
                evidence = response.choices[0].message["content"].strip()
                if evidence:
                    return {
                        "text": evidence,
                        "query": query,
                        "context": context or "",
                        "statement": atomic_statement or "",
                        "source": "generated",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

            except Mistral.APIError as e:
                logger.warning(f"OpenAI API error on attempt {attempt+1}/{num_retries}: {e}")
                if attempt < num_retries - 1:
                    time.sleep(2 * (attempt + 1))

        logger.error("Failed to generate evidence after all retries.")
        return {
            "text": "",
            "query": query,
            "context": context or "",
            "statement": atomic_statement or "",
            "error": "Failed after retries"
        }
    except Exception as e:
        logger.error(f"Unexpected error in evidence generation: {e}")
        return {
            "text": "",
            "query": query,
            "context": context or "",
            "statement": atomic_statement or "",
            "error": str(e)
        }

def batch_generate_evidence(
    questions: List[str],
    model: str,
    prompt: str,
    atomic_statements: Optional[List[str]] = None,
    current_statement_idx: Optional[int] = None
) -> List[Dict[str, str]]:
    try:
        context = None
        current_statement = None
        if atomic_statements and current_statement_idx is not None:
            if current_statement_idx > 0:
                context = " ".join(atomic_statements[:current_statement_idx])
            current_statement = atomic_statements[current_statement_idx]

        evidences = []
        for query in questions:
            ev = run_evidence_hallucination(
                query=query,
                model=model,
                prompt=prompt,
                context=context,
                atomic_statement=current_statement
            )
            evidences.append(ev)

        valid_evidences = [e for e in evidences if e.get("text") and not e.get("error")]
        if not valid_evidences:
            logger.warning("No valid evidences generated in this batch.")
        return valid_evidences

    except Exception as e:
        logger.error(f"Error in batch evidence generation: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    ...

