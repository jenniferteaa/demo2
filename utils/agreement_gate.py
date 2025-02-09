"""
agreement_gate.py

Agreement gate for fact verification and revision.
"""

import os
import time
import re
import json
import logging
from typing import Dict, Optional, Tuple, Any


import openai
import torch
from sentence_transformers import CrossEncoder
from openai.error import OpenAIError

from mistralai import Mistral
# Initialize Mistral API
api_key = "Replace with your valid API key"  # 
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# If you have a custom call_openai_api in utils, uncomment:
# from utils import call_openai_api

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize a CrossEncoder model to compute relevance scores between text pairs
RELEVANCE_SCORER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

def compute_relevance_score(query: str, evidence: str) -> float:
    """
    Computes a relevance score for (query, evidence) using the cross-encoder.
    Higher scores => more relevant.
    """
    try:
        with torch.no_grad():
            score = RELEVANCE_SCORER.predict([(query, evidence)])
        return float(score[0])
    except Exception as e:
        logger.error(f"Error computing relevance: {e}")
        return 0.0

def format_prompt(
    claim: str,
    query: str,
    evidence: str,
    context: Optional[str] = None,
    relevance_threshold: float = 0.1
) -> Tuple[str, bool]:
    """
    Builds the base prompt for analyzing whether 'claim' needs revision given 'evidence'.
    Also checks if evidence meets the `relevance_threshold`.
    Returns:
      - A textual prompt
      - A boolean indicating whether the evidence is relevant enough to consider.
    """
    # Compute relevance
    relevance = compute_relevance_score(query, evidence)
    logger.info(f"Evidence relevance score: {relevance:.2f}")

    is_relevant = (relevance >= relevance_threshold)

    # If not relevant, we don't include the context or evidence in the final prompt
    if not is_relevant:
        logger.info(f"Evidence ignored (low relevance: {relevance:.2f})")
        context = None
        # We might also blank out the evidence if we truly want to ignore it,
        # but let's keep it in the prompt for transparency. 
        # That said, the function's current approach doesn't explicitly remove it,
        # so you can do so if desired.

    # Build the basic instruction
    base_prompt = (
        "Task: Determine if the evidence contradicts or requires changes to the claim.\n"
        "Format: JSON with the following fields:\n"
        "- needs_revision: true/false\n"
        "- explanation: Short reason for or against revision\n"
        "- revised_claim: the updated text if revision is needed, else the same claim\n\n"
        f"Claim: {claim}\n"
        f"Query: {query}\n"
        f"Evidence: {evidence}\n"
    )

    if context:
        base_prompt = f"Context: {context}\n{base_prompt}"

    return base_prompt, is_relevant

def run_agreement_gate(
    claim: str,
    query: str,
    evidence: str,
    context: Optional[str] = None,
    model: str = "mistral-large-latest",
    temperature: float = 0.1,
    max_retries: int = 3,
    relevance_threshold: float = 0.1,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to determine if a claim needs revision based on the evidence.
    Steps:
      1. Format a JSON-based system prompt.
      2. Parse whether "needs_revision" is True/False.
      3. If True, run a second revision prompt to finalize the updated claim.
      4. Return a dict capturing the final status of the revision process.

    Returns a dict with:
        "is_open": bool, whether the gate requires revision
        "original_claim": str
        "revised_claim": str
        "explanation": str
        "context_used": bool
        "evidence_used": str
        "revision_score": float
    """
    try:
        # 1) Create a base prompt using `format_prompt`
        formatted_prompt, is_relevant = format_prompt(
            claim=claim,
            query=query,
            evidence=evidence,
            context=context,
            relevance_threshold=relevance_threshold
        )

        # If the user provided an additional prompt, prepend it
        if prompt:
            formatted_prompt = f"{prompt}\n\n{formatted_prompt}"

        # 2) Attempt multiple times in case of transient OpenAI errors
        for attempt in range(max_retries):
            try:
                response = client.chat.complete(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a fact-checking assistant analyzing evidence vs. claims. "
                                "Return JSON as instructed."
                            )
                        },
                        {
                            "role": "user",
                            "content": formatted_prompt
                        }
                    ],
                    temperature=temperature
                )
                #print("This is the raw response from agreement gate: ", response)
                result_raw = response.choices[0].message.content.strip()
                match = re.search(r'\{.*\}', result_raw, re.DOTALL)
                resultt = match.group(0) if match else None
                #print("THIS IS THE RESULT_RAW: ", resultt)

                logger.debug(f"Agreement gate raw response:\n{resultt}")

                # 3) Parse the JSON
                try:
                    result = json.loads(resultt)
                    #print("THIS IS THE RAW RESULT: ", result)
                except:
                    # If we can't parse JSON, assume no revision needed
                    logger.warning("Could not parse JSON from the response.")
                    result = {
                        "needs_revision": False,
                        "explanation": "No valid JSON returned",
                        "revised_claim": claim
                    }

                needs_revision = bool(result.get("needs_revision", False))
                explanation = result.get("explanation", "")
                revised_claim_in_result = result.get("revised_claim", "").strip()
                

                final_revised_claim = claim  # default to original
                if needs_revision:
                    # The model's answer might already have a "revised_claim",
                    # but we can refine with a second revision prompt:
                    revision_prompt = (
                        "You are a fact-checking assistant that revises claims based on evidence.\n"
                        f"Original claim: {claim}\n"
                        f"Evidence: {evidence}\n"
                        f"Reason for revision: {explanation}\n\n"
                        "Provide a concise, accurate revised claim."
                    )

                    revision_response = client.chat.complete(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a fact-checking assistant."},
                            {"role": "user", "content": revision_prompt},
                        ],
                        temperature=temperature
                    )
                    second_pass_claim = revision_response.choices[0].message.content.strip()
                    final_revised_claim = second_pass_claim or revised_claim_in_result or claim
                    #print("This is the final_revised_claim from agreement gate: ", final_revised_claim)
                # Compute a final relevance score for the revised claim
                revision_score = compute_relevance_score(final_revised_claim, evidence)

                return {
                    "is_open": needs_revision,
                    "original_claim": claim,
                    "revised_claim": final_revised_claim,
                    "explanation": explanation,
                    "context_used": is_relevant,
                    "evidence_used": evidence,
                    "revision_score": revision_score
                }

            except:
                logger.warning(f"OpenAI API error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    # Exponential or linear backoff; here we do exponential
                    time.sleep(2 ** attempt)

        # If we exhausted all attempts:
        raise Exception(f"Failed after {max_retries} attempts.")

    except Exception as e:
        logger.error(f"Agreement gate error: {e}")
        # If everything fails, return a fallback
        return {
            "is_open": False,
            "original_claim": claim,
            "revised_claim": claim,
            "explanation": str(e),
            "context_used": False,
            "evidence_used": evidence,
            "revision_score": 0.0
        }

if __name__ == "__main__":
    # Quick test
    test_claim = "The Wright brothers made their first flight in 1902."
    test_query = "When did the Wright brothers first achieve powered flight?"
    test_evidence = "They achieved their first powered, sustained flight on December 17, 1903."

    result = run_agreement_gate(
        claim=test_claim,
        query=test_query,
        evidence=test_evidence,
        model="mistral-large-latest",
        temperature=0.1,
        max_retries=2,
        relevance_threshold=0.1
    )
    print("This is the final result object: ", result)

    print("\n--- Agreement Gate Test ---")
    print(f"Original claim: {result['original_claim']}")
    print(f"Gate status: {'open' if result['is_open'] else 'closed'}")
    print(f"Revised claim: {result['revised_claim']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Evidence used: {result['evidence_used']}")
    print(f"Final revision score: {result['revision_score']:.3f}")
