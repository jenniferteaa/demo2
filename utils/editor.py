"""
editor.py (or utils for question generation and editing)

Utils for running question generation and editing with atomic statement processing.
"""
import os
import time
import re
from typing import List, Dict, Any, Optional
from mistralai import Mistral

import openai

# If you have a custom call_openai_api, you can import it here:
# from utils import call_openai_api

# Set your OpenAI API key
#openai.api_key = os.getenv("OPENAI_API_KEY")

api_key = "Replace with your valid API key"  # 
model = "mistral-large-latest"
client = Mistral(api_key=api_key)
# --------------------------------------------------------------------------
# 1. SPLIT TEXT INTO ATOMIC STATEMENTS
# --------------------------------------------------------------------------
def split_into_atomic_statements(text: str) -> List[str]:
    """
    Splits a paragraph into atomic statements using punctuation heuristics.
    Primarily looks for sentence-ending punctuation (periods, exclamation marks, question marks).
    
    Returns:
      A list of statements (strings), each focusing on a single fact or claim.
    """
    if not text or not text.strip():
        return []
    
    # Clean up repeated punctuation
    text = text.replace("..", ".").replace("...", ".")
    text = text.replace("!.", "!").replace("?.", "?")

    # Split on recognized sentence endings. This regex:
    #   - looks behind for a punctuation mark (period, exclamation, question),
    #   - then looks ahead for a capital letter or end-of-string,
    #   - splits on the whitespace boundary
    # This helps keep punctuation with the preceding sentence.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', text)

    atomic_statements = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Ensure sentence ends with punctuation
            if sentence[-1] not in ".!?":
                sentence += "."
            # Capitalize the first character (if alpha)
            if sentence[0].isalpha():
                sentence = sentence[0].upper() + sentence[1:]
            atomic_statements.append(sentence)
    return atomic_statements


# --------------------------------------------------------------------------
# 2. CONTEXT MANAGEMENT
# --------------------------------------------------------------------------
def maintain_context(statements: List[str], current_index: int) -> str:
    """
    Build a "context" string from all previously processed statements,
    so each subsequent statement can reference prior content if needed.
    """
    if current_index == 0:
        return ""
    # Join all statements up to (but not including) the current index
    return " ".join(statements[:current_index])


# --------------------------------------------------------------------------
# 3. QUESTION GENERATION
# --------------------------------------------------------------------------
def parse_question_api_response(api_response: str) -> List[str]:
    """
    Parse lines that begin with a digit from the GPT response as questions.
    E.g., "1. When was X founded?"
    """
    questions = []
    for line in api_response.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Extract everything after the digit and punctuation
            space_index = line.find(" ")
            if space_index != -1:
                question = line[space_index:].strip()
                question = re.sub(r'^[.)\s]+', '', question)
                if question:
                    questions.append(question)
    return questions

def run_rarr_question_generation(
    claim: str,
    model: str,
    prompt: str,
    temperature: float,
    num_questions: int = 3,
    context: Optional[str] = None,
    num_retries: int = 3,
) -> List[str]:
    """
    Generates exactly `num_questions` questions for the claim to verify facts.
    If `context` is given, it can be included in the prompt for continuity.
    
    If you want to use call_openai_api, replace the direct openai.ChatCompletion.create calls
    with something like:
    
        response = call_openai_api(
            messages=[...],
            model=model,
            temperature=temperature,
            max_retries=1
        )
    """
    # Customize the prompt to force exactly num_questions
    question_instruction = (
        f"\nPlease generate exactly {num_questions} specific, unique questions "
        "focusing on verifiable facts."
    )
    base_prompt = prompt + question_instruction

    # Always provide a string for `context` to avoid KeyError
    user_prompt = base_prompt.format(claim=claim, context=context or "").strip()

    questions = set()
    attempts = 0

    while len(questions) < num_questions and attempts < num_retries:
        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fact-checking assistant generating questions."
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=256,
            )
            new_questions = parse_question_api_response(
                response.choices[0].message["content"]
            )
            for q in new_questions:
                #print("This is the question in editor's run_rarr_question_generator function: ", q)
                questions.add(q)
        except Mistral.APIError:
            time.sleep(1.5 * (attempts + 1))
        attempts += 1

    # If fewer than requested, pad with fallback questions
    questions_list = list(questions)[:num_questions]
    while len(questions_list) < num_questions:
        questions_list.append(f"What evidence supports the claim that '{claim}'?")

    return questions_list


# --------------------------------------------------------------------------
# 4. EDITOR LOGIC
# --------------------------------------------------------------------------
def parse_editor_response(api_response: str) -> Optional[str]:
    """
    Attempt to parse a revised text from the GPT response.
    Looks for line starting with "My fix:" or a well-formed sentence.
    """
    api_response_lines = api_response.strip().split("\n")
    # Look for "My fix:"
    for line in api_response_lines:
        if "My fix:" in line:
            edited_claim = line.split("My fix:")[-1].strip()
            if edited_claim:
                return edited_claim
    
    # Fallback: look for a likely-sentence line
    for line in api_response_lines:
        line = line.strip()
        if line and line[0].isupper() and line[-1] in '.!?':
            return line
    return None

def run_rarr_editor(
    claim: str,
    query: str,
    evidence: str,
    model: str,
    prompt: str,
    context: Optional[str] = None,
    temperature: float = 0.0,
    num_retries: int = 3
) -> Dict[str, str]:
    """
    Runs an editor that modifies the `claim` based on the `query` and `evidence`.
    - `context` can include previous statements for continuity.
    
    If you want to use call_openai_api, replace the direct openai.ChatCompletion.create calls
    with something like:
    
        response = call_openai_api(
            messages=[...],
            model=model,
            temperature=temperature,
            max_retries=1
        )
    """
    user_content = prompt.format(
        claim=claim,
        query=query,
        evidence=evidence,
        context=context or ""
    )

    attempt = 0
    while attempt < num_retries:
        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise editor that makes factual corrections "
                            "based on evidence while preserving the original claim's style."
                        )
                    },
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            #response.choices[0].message.content.strip()
            output_text = response.choices[0].message.content
            edited_claim = parse_editor_response(output_text)

            if edited_claim:
                # Basic normalization
                edited_claim = edited_claim.strip()
                if edited_claim and edited_claim[0].isalpha():
                    edited_claim = edited_claim[0].upper() + edited_claim[1:]
                if edited_claim and edited_claim[-1] not in ".!?":
                    edited_claim += "."
                return {"text": edited_claim}

        except:
            print("The except block in editor's run_rarr_editor was executed")
            time.sleep(2 * (attempt + 1))
        attempt += 1

    # If all retries fail, return the original
    return {"text": claim}


# --------------------------------------------------------------------------
# 5. ORCHESTRATOR EXAMPLE: process_paragraph
# --------------------------------------------------------------------------
def process_paragraph(
    paragraph: str,
    model: str,
    question_prompt: str,
    editor_prompt: str,
    temperature: float = 0.7,
    num_questions: int = 3,
) -> Dict[str, Any]:
    """
    A simplified demonstration of:
      1) Splitting `paragraph` into atomic statements
      2) Generating questions for each statement
      3) Attempting to revise each statement using an editor
      4) Combining them back into a final revised text

    **Note**: This example sets `evidence=""` because actual evidence would come from
    your search/evidence_selection modules in a full pipeline.
    """
    statements = split_into_atomic_statements(paragraph)
    results = {
        "original_statements": statements,
        "questions_per_statement": [],
        "revised_statements": [],
        "intermediate_steps": [],
        "final_text": ""
    }

    for i, statement in enumerate(statements):
        # Gather context from previously revised statements
        context = " ".join(results["revised_statements"]) if i > 0 else ""

        # 1) Generate questions
        questions = run_rarr_question_generation(
            claim=statement,
            model=model,
            prompt=question_prompt,
            temperature=temperature,
            num_questions=num_questions,
            context=context
        )
        results["questions_per_statement"].append(questions)

        # 2) Initialize the current revision as the original statement
        current_revision = statement

        # 3) For each question, attempt to revise the statement
        for q in questions:
            # In a full pipeline, you'd replace `""` with real evidence
            evidence = ""
            edited_result = run_rarr_editor(
                claim=current_revision,
                query=q,
                evidence=evidence,
                model=model,
                prompt=editor_prompt,
                context=context,
                temperature=0.0
            )
            # If there's a meaningful change, update the current revision
            if edited_result["text"] != current_revision:
                current_revision = edited_result["text"]
                results["intermediate_steps"].append({
                    "statement": statement,
                    "question": q,
                    "revision": current_revision
                })

        # 4) Add the final revised statement to the list
        results["revised_statements"].append(current_revision)

    # Combine all revised statements into a single final text
    results["final_text"] = " ".join(results["revised_statements"])
    return results


# --------------------------------------------------------------------------
# EXAMPLE USAGE (if running this file directly)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Example paragraph
    example_paragraph = (
        "Lena Headey portrayed Cersei Lannister in HBO's Game of Thrones since 2011. "
        "She received multiple Emmy nominations for this role. "
        "She became one of the highest-paid television actors by 2017"
    )

    # Example prompts
    question_prompt = (
        "Given the claim: '{claim}'\n"
        "Context: '{context}'\n"
        "Generate specific questions to verify factual accuracy."
    )
    editor_prompt = (
        "Original claim: {claim}\n"
        "Query: {query}\n"
        "Evidence: {evidence}\n"
        "Context: {context}\n"
        "My fix:"
    )

    results = process_paragraph(
        paragraph=example_paragraph,
        model="mistral-large-latest",
        question_prompt=question_prompt,
        editor_prompt=editor_prompt,
        temperature=0.7,
        num_questions=3
    )

    print("\n--- Results ---")
    print("Original statements:")
    for s in results["original_statements"]:
        print(f" - {s}")
    print("\nQuestions per statement:")
    for i, qs in enumerate(results["questions_per_statement"], start=1):
        print(f"Statement {i} questions: {qs}")
    print("\nRevised statements:")
    for rs in results["revised_statements"]:
        print(f" - {rs}")
    print("\nIntermediate steps:")
    for step in results["intermediate_steps"]:
        print(step)
    print("\nFinal text:")
    print(results["final_text"])
