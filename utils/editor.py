"""
editor.py (or utils for question generation and editing)

Utils for running question generation and editing with atomic statement processing.
"""
import os
import time
import re
from typing import List, Dict, Any, Optional

import openai

# If you have a custom call_openai_api, you can import it here:
# from utils import call_openai_api

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


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
            response = openai.ChatCompletion.create(
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
                questions.add(q)
        except openai.error.OpenAIError:
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
            response = openai.ChatCompletion.create(
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
            output_text = response.choices[0].message["content"]
            edited_claim = parse_editor_response(output_text)

            if edited_claim:
                # Basic normalization
                edited_claim = edited_claim.strip()
                if edited_claim and edited_claim[0].isalpha():
                    edited_claim = edited_claim[0].upper() + edited_claim[1:]
                if edited_claim and edited_claim[-1] not in ".!?":
                    edited_claim += "."
                return {"text": edited_claim}

        except openai.error.OpenAIError:
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
        model="gpt-3.5-turbo",
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



# """
# Utils for running question generation and editing with atomic statement processing.
# """
# import os
# import time
# import re
# from typing import List, Dict, Union, Any, Tuple, Optional

# import openai

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")


# # --------------------------------------------------------------------------
# # 1. SPLIT TEXT INTO ATOMIC STATEMENTS
# # --------------------------------------------------------------------------
# def split_into_atomic_statements(text: str) -> List[str]:
#     """
#     Splits a paragraph into atomic statements using punctuation heuristics.
#     Primarily looks for sentence-ending punctuation (periods, exclamation marks, question marks).
    
#     Returns:
#       A list of statements (strings), each focusing on a single fact or claim.
#     """
#     if not text or not text.strip():
#         return []
    
#     # Clean up repeated punctuation
#     text = text.replace("..", ".").replace("...", ".")
#     text = text.replace("!.", "!").replace("?.", "?")

#     # Split on recognized sentence endings. This regex:
#     #   - looks behind for a punctuation mark (period, exclamation, question),
#     #   - then looks ahead for a capital letter or end-of-string,
#     #   - splits on the whitespace boundary
#     # This helps keep punctuation with the preceding sentence.
#     sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', text)

#     atomic_statements = []
#     for sentence in sentences:
#         # Clean and normalize
#         sentence = sentence.strip()
#         if sentence:
#             # Ensure sentence ends with punctuation
#             if sentence[-1] not in ".!?":
#                 sentence += "."
#             # Capitalize the first character (if alpha)
#             if sentence[0].isalpha():
#                 sentence = sentence[0].upper() + sentence[1:]
#             atomic_statements.append(sentence)

#     return atomic_statements


# # --------------------------------------------------------------------------
# # 2. CONTEXT MANAGEMENT
# # --------------------------------------------------------------------------
# def maintain_context(statements: List[str], current_index: int) -> str:
#     """
#     Build a "context" string from all previously processed statements,
#     so each subsequent statement can reference prior content if needed.
#     """
#     if current_index == 0:
#         return ""
#     # Join all statements up to (but not including) the current index
#     return " ".join(statements[:current_index])


# # --------------------------------------------------------------------------
# # 3. QUESTION GENERATION (BASIC)
# # --------------------------------------------------------------------------
# def parse_question_api_response(api_response: str) -> List[str]:
#     """
#     Parse lines that begin with a digit from the GPT response as questions.
#     E.g., "1. When was X founded?"
#     """
#     questions = []
#     for line in api_response.split("\n"):
#         line = line.strip()
#         if line and line[0].isdigit():
#             # Extract everything after the digit and punctuation
#             space_index = line.find(" ")
#             if space_index != -1:
#                 question = line[space_index:].strip()
#                 question = re.sub(r'^[.)\s]+', '', question)
#                 if question:
#                     questions.append(question)
#     return questions

# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     prompt: str,
#     temperature: float,
#     num_questions: int = 3,
#     context: Optional[str] = None,
#     num_retries: int = 3,
# ) -> List[str]:
#     """
#     Generates exactly `num_questions` questions for the claim to verify facts.
#     If `context` is given, it can be included in the prompt for continuity.
#     """
#     # Customize prompt to force exactly num_questions
#     question_instruction = (
#         f"\nPlease generate exactly {num_questions} specific, unique questions "
#         "focusing on verifiable facts."
#     )
#     # Merge prompt with the question instruction
#     base_prompt = prompt + question_instruction

#     if context:
#         # Provide context in the user prompt if needed
#         user_prompt = base_prompt.format(context=context, claim=claim).strip()
#     else:
#         user_prompt = base_prompt.format(claim=claim).strip()

#     questions = set()
#     attempts = 0

#     while len(questions) < num_questions and attempts < num_retries:
#         try:
#             response = openai.ChatCompletion.create(
#                 model=model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a fact-checking assistant generating questions."
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 temperature=temperature,
#                 max_tokens=256,
#             )
#             new_questions = parse_question_api_response(response.choices[0].message["content"])
#             for q in new_questions:
#                 questions.add(q)
#         except openai.error.OpenAIError as e:
#             time.sleep(1.5 * (attempts + 1))
#         attempts += 1

#     # If fewer than requested, pad with fallback questions
#     questions_list = list(questions)[:num_questions]
#     while len(questions_list) < num_questions:
#         questions_list.append(f"What evidence supports the claim that '{claim}'?")

#     return questions_list


# # --------------------------------------------------------------------------
# # 4. EDITOR LOGIC
# # --------------------------------------------------------------------------
# def parse_editor_response(api_response: str) -> Optional[str]:
#     """
#     Attempt to parse a revised text from the GPT response.
#     Looks for line starting with something like "My fix:" or a well-formed sentence.
#     """
#     api_response_lines = api_response.strip().split("\n")
#     # Look for "My fix:"
#     for line in api_response_lines:
#         if "My fix:" in line:
#             # Return everything after "My fix:"
#             edited_claim = line.split("My fix:")[-1].strip()
#             if edited_claim:
#                 return edited_claim
    
#     # Fallback: look for a likely-sentence line
#     for line in api_response_lines:
#         line = line.strip()
#         if line and line[0].isupper() and line[-1] in '.!?':
#             return line

#     return None

# def run_rarr_editor(
#     claim: str,
#     query: str,
#     evidence: str,
#     model: str,
#     prompt: str,
#     context: Optional[str] = None,
#     temperature: float = 0.0,
#     num_retries: int = 3
# ) -> Dict[str, str]:
#     """
#     Runs an editor that modifies the `claim` based on the `query` and `evidence`.
#     - `context` can include previous statements for continuity.
#     """
#     # Build the user prompt
#     if context:
#         user_content = prompt.format(context=context, claim=claim, query=query, evidence=evidence)
#     else:
#         user_content = prompt.format(claim=claim, query=query, evidence=evidence)

#     attempt = 0
#     while attempt < num_retries:
#         try:
#             response = openai.ChatCompletion.create(
#                 model=model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": (
#                             "You are a precise editor that makes factual corrections "
#                             "based on evidence while preserving the original claim's style."
#                         )
#                     },
#                     {"role": "user", "content": user_content},
#                 ],
#                 temperature=temperature,
#                 max_tokens=512,
#             )
#             output_text = response.choices[0].message["content"]
#             edited_claim = parse_editor_response(output_text)

#             if edited_claim:
#                 # Basic normalization
#                 edited_claim = edited_claim.strip()
#                 if edited_claim and edited_claim[0].isalpha():
#                     edited_claim = edited_claim[0].upper() + edited_claim[1:]
#                 if edited_claim and edited_claim[-1] not in ".!?":
#                     edited_claim += "."
#                 return {"text": edited_claim}

#         except openai.error.OpenAIError:
#             time.sleep(2 * (attempt + 1))
#         attempt += 1

#     # If all retries fail, return the original
#     return {"text": claim}


# # --------------------------------------------------------------------------
# # 5. ORCHESTRATOR EXAMPLE: process_paragraph
# # --------------------------------------------------------------------------
# def process_paragraph(
#     paragraph: str,
#     model: str,
#     question_prompt: str,
#     editor_prompt: str,
#     temperature: float = 0.7,
#     num_questions: int = 3,
# ) -> Dict[str, Any]:
#     """
#     A simplified demonstration of:
#       1) Splitting `paragraph` into atomic statements
#       2) Generating questions for each statement
#       3) Attempting to revise each statement using an editor
#       4) Combining them back into a final revised text

#     **Note**: This example sets `evidence=""` because actual evidence would come from
#     your search/evidence_selection modules in a full pipeline.
#     """
#     statements = split_into_atomic_statements(paragraph)
#     results = {
#         "original_statements": statements,
#         "questions_per_statement": [],
#         "revised_statements": [],
#         "intermediate_steps": [],
#         "final_text": ""
#     }

#     for i, statement in enumerate(statements):
#         # Gather context from previously revised statements
#         context = " ".join(results["revised_statements"]) if i > 0 else ""

#         # 1) Generate questions
#         questions = run_rarr_question_generation(
#             claim=statement,
#             model=model,
#             prompt=question_prompt,
#             temperature=temperature,
#             num_questions=num_questions,
#             context=context
#         )
#         results["questions_per_statement"].append(questions)

#         # 2) Initialize the current revision as the original statement
#         current_revision = statement

#         # 3) For each question, attempt to revise the statement
#         for q in questions:
#             # In a full RARR pipeline, replace the empty string with real evidence
#             evidence = ""
#             edited_result = run_rarr_editor(
#                 claim=current_revision,
#                 query=q,
#                 evidence=evidence,
#                 model=model,
#                 prompt=editor_prompt,
#                 context=context,
#                 temperature=0.0
#             )
#             # If there's a meaningful change, update the current revision
#             if edited_result["text"] != current_revision:
#                 current_revision = edited_result["text"]
#                 results["intermediate_steps"].append({
#                     "statement": statement,
#                     "question": q,
#                     "revision": current_revision
#                 })

#         # 4) Add the final revised statement to the list
#         results["revised_statements"].append(current_revision)

#     # Combine all revised statements into a single final text
#     results["final_text"] = " ".join(results["revised_statements"])
#     return results


# # --------------------------------------------------------------------------
# # EXAMPLE USAGE (if running this file directly)
# # --------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Example paragraph
#     example_paragraph = (
#         "Lena Headey portrayed Cersei Lannister in HBO's Game of Thrones since 2011. "
#         "She received multiple Emmy nominations for this role. "
#         "She became one of the highest-paid television actors by 2017"
#     )

#     # Example prompts
#     question_prompt = (
#         "Given the claim: '{claim}'\n"
#         "Context: '{context}'\n"
#         "Generate specific questions to verify factual accuracy."
#     )
#     editor_prompt = (
#         "Original claim: {claim}\n"
#         "Query: {query}\n"
#         "Evidence: {evidence}\n"
#         "Context: {context}\n"
#         "My fix:"
#     )

#     results = process_paragraph(
#         paragraph=example_paragraph,
#         model="gpt-3.5-turbo",
#         question_prompt=question_prompt,
#         editor_prompt=editor_prompt,
#         temperature=0.7,
#         num_questions=3
#     )

#     print("\n--- Results ---")
#     print("Original statements:")
#     for s in results["original_statements"]:
#         print(f" - {s}")
#     print("\nQuestions per statement:")
#     for i, qs in enumerate(results["questions_per_statement"], start=1):
#         print(f"Statement {i} questions: {qs}")
#     print("\nRevised statements:")
#     for rs in results["revised_statements"]:
#         print(f" - {rs}")
#     print("\nIntermediate steps:")
#     for step in results["intermediate_steps"]:
#         print(step)
#     print("\nFinal text:")
#     print(results["final_text"])


# """Utils for running question generation and editing."""
# import os
# import time
# from typing import List, Dict, Union, Any

# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")

# def parse_api_response(api_response: str) -> List[str]:
#     """Extract questions from the GPT API response.

#     Args:
#         api_response: Question generation response from GPT.
#     Returns:
#         questions: A list of questions.
#     """
#     questions = []
#     for line in api_response.split("\n"):
#         line = line.strip()
#         if line and line[0].isdigit():
#             # Extract question text after the number and optional period/parenthesis
#             question = line[line.find(" ") + 1:].strip()
#             questions.append(question)
#     return questions

# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     prompt: str,
#     temperature: float,
#     num_rounds: int,
#     context: str = None,
#     num_retries: int = 5,
# ) -> List[str]:
#     """Generates questions that interrogate the information in a claim.

#     Args:
#         claim: Text to generate questions off of.
#         model: Name of the OpenAI GPT model to use.
#         prompt: The prompt template to query GPT with.
#         temperature: Temperature to use for sampling questions. 0 represents greedy decoding.
#         num_rounds: Number of times to sample questions.
#         context: Optional context to include with the claim.
#         num_retries: Number of retries in case of API errors.
#     Returns:
#         questions: A list of unique questions.
#     """
#     if context:
#         gpt_input = prompt.format(context=context, claim=claim).strip()
#     else:
#         gpt_input = prompt.format(claim=claim).strip()

#     questions = set()
#     for _ in range(num_rounds):
#         for _ in range(num_retries):
#             try:
#                 response = openai.ChatCompletion.create(
#                     model=model,
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant."},
#                         {"role": "user", "content": gpt_input},
#                     ],
#                     temperature=temperature,
#                     max_tokens=256,
#                 )
#                 cur_round_questions = parse_api_response(response.choices[0].message["content"].strip())
#                 questions.update(cur_round_questions)
#                 break
#             except openai.error.OpenAIError as exception:
#                 print(f"{exception}. Retrying...")
#                 time.sleep(1)

#     return list(sorted(questions))

# def parse_editor_response(api_response: str) -> str:
#     """Extract the edited text from the GPT API response.

#     Args:
#         api_response: Editor response from GPT.
#     Returns:
#         edited_claim: The edited claim.
#     """
#     api_response = api_response.strip().split("\n")
#     if len(api_response) < 2:
#         print("Editor error.")
#         return None
#     edited_claim = api_response[1].split("My fix:")[-1].strip()
#     return edited_claim

# def run_rarr_editor(
#     claim: str,
#     query: str,
#     evidence: str,
#     model: str,
#     prompt: str,
#     context: str = None,
#     num_retries: int = 5,
# ) -> Dict[str, str]:
#     """Runs a GPT editor on the claim given a query and evidence to support the edit.

#     Args:
#         claim: Text to edit.
#         query: Query to guide the editing.
#         evidence: Evidence to base the edit on.
#         model: Name of the OpenAI GPT model to use.
#         prompt: The prompt template to query GPT with.
#         context: Optional context to include with the claim.
#         num_retries: Number of times to retry OpenAI call in the event of an API failure.
#     Returns:
#         output: A dictionary containing the edited claim.
#     """
#     if context:
#         gpt_input = prompt.format(
#             context=context, claim=claim, query=query, evidence=evidence
#         ).strip()
#     else:
#         gpt_input = prompt.format(claim=claim, query=query, evidence=evidence).strip()

#     for _ in range(num_retries):
#         try:
#             response = openai.ChatCompletion.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": gpt_input},
#                 ],
#                 temperature=0.0,
#                 max_tokens=512,
#             )
#             output_text = response.choices[0].message["content"]
#             edited_claim = parse_editor_response(output_text)
#             # If there was an error in GPT generation, return the original claim.
#             if not edited_claim:
#                 edited_claim = claim
#             return {"text": edited_claim}

#         except openai.error.OpenAIError as exception:
#             print(f"{exception}. Retrying...")
#             time.sleep(2)

#     # Return original claim if retries are exhausted
#     return {"text": claim}








# """Utils for running the editor."""
# import os
# import time
# from typing import Dict, Union

# import openai

# # Set up OpenAI API key from environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def parse_api_response(api_response: str) -> Union[str, None]:
#     """Extract the edited claim from the API response.

#     The prompt returns a reason for the edit and the actual edit in two consecutive lines.
#     This function only extracts the edit from the second line.

#     Args:
#         api_response: Editor response from the model.
#     Returns:
#         edited_claim: The edited claim.
#     """
#     api_response_lines = api_response.strip().split("\n")
#     if len(api_response_lines) < 2:
#         print("Editor error: Incomplete response.")
#         return None
#     edited_claim = api_response_lines[1].split("My fix:")[-1].strip()
#     return edited_claim

# def run_rarr_editor(
#     claim: str,
#     query: str,
#     evidence: str,
#     model: str,
#     prompt: str,
#     context: str = None,
#     num_retries: int = 5,
# ) -> Dict[str, str]:
#     """Runs the editor on the claim, using the query and evidence to guide the edit.

#     Args:
#         claim: Text to edit.
#         query: Query to guide the editing.
#         evidence: Evidence to base the edit on.
#         model: Name of the OpenAI model to use (e.g., "gpt-3.5-turbo" or "text-davinci-003").
#         prompt: The prompt template to query the model with.
#         context: Optional context to include with the claim.
#         num_retries: Number of times to retry the OpenAI call in case of API failure.
#     Returns:
#         output: A dictionary with the edited claim text.
#     """
#     # Format the input prompt based on whether context is provided
#     if context:
#         formatted_input = prompt.format(
#             context=context, claim=claim, query=query, evidence=evidence
#         ).strip()
#     else:
#         formatted_input = prompt.format(claim=claim, query=query, evidence=evidence).strip()

#     for _ in range(num_retries):
#         try:
#             # Check if the model is a chat model (like gpt-3.5-turbo) and use ChatCompletion
#             if "gpt-3.5-turbo" in model or "gpt-4" in model:
#                 response = openai.ChatCompletion.create(
#                     model=model,
#                     messages=[{"role": "user", "content": formatted_input}],
#                     temperature=0.0,
#                     max_tokens=512,
#                 )
#                 response_text = response.choices[0].message["content"]
#             else:
#                 # Use regular Completion endpoint for non-chat models
#                 response = openai.Completion.create(
#                     model=model,
#                     prompt=formatted_input,
#                     temperature=0.0,
#                     max_tokens=512,
#                     stop=["\n\n"],
#                 )
#                 response_text = response.choices[0].text

#             # Parse the response and break the retry loop
#             edited_claim = parse_api_response(response_text)
#             break

#         except openai.error.OpenAIError as exception:
#             print(f"{exception}. Retrying...")
#             time.sleep(2)

#     # If there was an error in GPT-3 generation, return the original claim
#     if not edited_claim:
#         edited_claim = claim

#     output = {"text": edited_claim}
#     return output



# """Utils for running the editor."""
# import os
# import time
# import logging
# from typing import Dict, Union, Any
# import openai

# # Setup logging configuration with detailed output
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def setup_openai_client():
#     """Sets up the OpenAI client with API key validation."""
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable is not set")

#     if hasattr(openai, 'OpenAI'):
#         return openai.OpenAI(api_key=api_key)
#     else:
#         openai.api_key = api_key
#         return openai

# # Initialize the OpenAI client
# client = setup_openai_client()

# def parse_api_response(api_response: str) -> str:
#     """Extracts the edited claim from the GPT response.
    
#     Args:
#         api_response: Raw response string from GPT.
        
#     Returns:
#         The edited claim, or None if parsing fails.
#     """
#     try:
#         response_lines = api_response.strip().split("\n")
#         edited_claim = response_lines[0].strip()  # Take the first line as edited claim
#         return edited_claim if edited_claim else None
#     except Exception as e:
#         logging.error(f"Editor error in parsing response: {str(e)}")
#         return None

# def run_rarr_editor(
#     claim: str,
#     query: str,
#     evidence: str,
#     model: str,
#     prompt: str,
#     context: str = None,
#     num_retries: int = 5,
#     client: Any = None,
# ) -> Dict[str, str]:
#     """Runs a GPT-based editor on the claim using query and evidence.

#     Args:
#         claim: Original text to be edited.
#         query: Query guiding the edit.
#         evidence: Evidence supporting the edit.
#         model: Model name for OpenAI.
#         prompt: Prompt template for the GPT model.
#         context: Optional context for enhancing the prompt.
#         num_retries: Maximum retries in case of API failure.
#         client: OpenAI client instance.
        
#     Returns:
#         A dictionary with the edited claim.
#     """
#     if client is None:
#         client = setup_openai_client()

#     # Enhanced prompt for better evidence incorporation
#     gpt3_input = f"""Carefully edit the claim to align with the evidence while preserving essential information. 
# Make specific, factual edits based on the evidence provided. Keep the original structure where possible,
# but ensure all statements are supported by the evidence.

# Context (if available): {context or ""}

# Original Claim: {claim}

# Evidence to consider: {evidence}

# Query to address: {query}

# Instructions:
# 1. Make precise edits to align with evidence
# 2. Keep original meaning where supported
# 3. Add specific details from evidence
# 4. Ensure factual accuracy

# Edited claim:"""

#     total_time = 0
#     for attempt in range(1, num_retries + 1):
#         start_time = time.time()

#         try:
#             if hasattr(client, 'chat'):
#                 response = client.chat.completions.create(
#                     model=model,
#                     messages=[{
#                         "role": "system",
#                         "content": "You are a precise editor focused on factual accuracy and evidence-based editing."
#                     }, {
#                         "role": "user",
#                         "content": gpt3_input
#                     }],
#                     temperature=0.0,
#                     max_tokens=1024,
#                     presence_penalty=0.6,  # Encourage new information
#                     frequency_penalty=0.3,  # Reduce repetition
#                     top_p=0.9  # Focus on most likely tokens
#                 )
#                 api_response = response.choices[0].message.content
#             else:
#                 response = client.Completion.create(
#                     model=model,
#                     prompt=gpt3_input,
#                     temperature=0.0,
#                     max_tokens=1024,
#                     presence_penalty=0.6,
#                     frequency_penalty=0.3,
#                     top_p=0.9,
#                     stop=["\n\n"]
#                 )
#                 api_response = response.choices[0].text

#             end_time = time.time()
#             total_time += end_time - start_time

#             # Parse and validate the response
#             edited_claim = parse_api_response(api_response)
#             if edited_claim:
#                 logging.info(f"Successfully edited claim in {end_time - start_time:.2f} seconds")
#                 logging.info(f"Original length: {len(claim)}, Edited length: {len(edited_claim)}")
#                 return {"text": edited_claim}
#             else:
#                 logging.warning("Failed to parse editor response. Retrying...")

#         except Exception as e:
#             end_time = time.time()
#             total_time += end_time - start_time
#             logging.warning(f"Attempt {attempt} failed: {str(e)}. Retrying in 2 seconds...")
#             time.sleep(2)

#     # If all attempts fail, return original claim
#     logging.error(f"Failed to edit claim after {num_retries} attempts, total time: {total_time:.2f} seconds")
#     return {"text": claim}


# # """Utils for running the editor."""
# # import os
# # import time
# # import logging
# # from typing import Dict, Union, Any
# # import openai

# # # Setup logging configuration with detailed output
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # def setup_openai_client():
# #     """Sets up the OpenAI client with API key validation."""
# #     api_key = os.getenv("OPENAI_API_KEY")
# #     if not api_key:
# #         raise ValueError("OPENAI_API_KEY environment variable is not set")

# #     if hasattr(openai, 'OpenAI'):
# #         return openai.OpenAI(api_key=api_key)
# #     else:
# #         openai.api_key = api_key
# #         return openai

# # # Initialize the OpenAI client
# # client = setup_openai_client()

# # def parse_api_response(api_response: str) -> str:
# #     """Extracts the edited claim from the GPT-3 API response.
    
# #     Args:
# #         api_response: The raw response string from GPT-3.
        
# #     Returns:
# #         The edited claim, or None if parsing fails.
# #     """
# #     response_lines = api_response.strip().split("\n")
# #     if len(response_lines) < 2:
# #         logging.error("Editor error: Insufficient lines in response.")
# #         return None
# #     # Extract and return the edited claim
# #     edited_claim = response_lines[1].split("My fix:")[-1].strip()
# #     return edited_claim

# # def run_rarr_editor(
# #     claim: str,
# #     query: str,
# #     evidence: str,
# #     model: str,
# #     prompt: str,
# #     context: str = None,
# #     num_retries: int = 5,
# #     client: Any = None,
# # ) -> Dict[str, str]:
# #     """Runs a GPT-based editor on the claim using query and evidence.

# #     Args:
# #         claim: Original text to be edited.
# #         query: Query guiding the edit.
# #         evidence: Evidence supporting the edit.
# #         model: Model name for OpenAI.
# #         prompt: Prompt template for the GPT model.
# #         context: Optional context for enhancing the prompt.
# #         num_retries: Maximum retries in case of API failure.
# #         client: OpenAI client instance.
        
# #     Returns:
# #         A dictionary with the edited claim.
# #     """
# #     if client is None:
# #         client = setup_openai_client()

# #     gpt3_input = prompt.format(
# #         context=context or "", claim=claim, query=query, evidence=evidence
# #     ).strip()

# #     # Timing and retry counters for performance analysis
# #     total_time, attempts = 0, 0

# #     for attempt in range(1, num_retries + 1):
# #         attempts += 1
# #         start_time = time.time()

# #         try:
# #             # API request based on the OpenAI client version
# #             if hasattr(client, 'chat'):
# #                 response = client.chat.completions.create(
# #                     model=model,
# #                     messages=[{"role": "user", "content": gpt3_input}],
# #                     temperature=0.0,
# #                     max_tokens=1024,
# #                     stop=["\n\n"],
# #                 )
# #                 api_response = response.choices[0].message.content
# #             else:
# #                 response = client.Completion.create(
# #                     model=model,
# #                     prompt=gpt3_input,
# #                     temperature=0.0,
# #                     max_tokens=512,
# #                     stop=["\n\n"],
# #                 )
# #                 api_response = response.choices[0].text

# #             end_time = time.time()
# #             total_time += end_time - start_time  # Add elapsed time for this attempt

# #             # Parse and return the response if successful
# #             edited_claim = parse_api_response(api_response)
# #             if edited_claim:
# #                 logging.info(f"Attempt {attempt} successful in {end_time - start_time:.2f} seconds.")
# #                 return {"text": edited_claim}
# #             else:
# #                 logging.warning("Failed to parse editor response. Returning original claim.")
# #                 return {"text": claim}

# #         except Exception as e:
# #             logging.warning(f"Attempt {attempt} failed: {str(e)}. Retrying in 2 seconds...")
# #             time.sleep(2)

# #     # Log total retry time if all attempts fail
# #     logging.error(f"Failed after {num_retries} attempts, total time: {total_time:.2f} seconds")
# #     return {"text": claim}  # Return the original claim if all attempts fail



# # """Utils for running the editor."""
# # import os
# # import time
# # from typing import Dict, Union

# # import openai

# # openai.api_key = os.getenv("OPENAI_API_KEY")


# # def parse_api_response(api_response: str) -> str:
# #     """Extract the agreement gate state and the reasoning from the GPT-3 API response.

# #     Our prompt returns a reason for the edit and the edit in two consecutive lines.
# #     Only extract out the edit from the second line.

# #     Args:
# #         api_response: Editor response from GPT-3.
# #     Returns:
# #         edited_claim: The edited claim.
# #     """
# #     api_response = api_response.strip().split("\n")
# #     if len(api_response) < 2:
# #         print("Editor error.")
# #         return None
# #     edited_claim = api_response[1].split("My fix:")[-1].strip()
# #     return edited_claim


# # def run_rarr_editor(
# #     claim: str,
# #     query: str,
# #     evidence: str,
# #     model: str,
# #     prompt: str,
# #     context: str = None,
# #     num_retries: int = 5,
# # ) -> Dict[str, str]:
# #     """Runs a GPT-3 editor on the claim given a query and evidence to support the edit.

# #     Args:
# #         claim: Text to edit.
# #         query: Query to guide the editing.
# #         evidence: Evidence to base the edit on.
# #         model: Name of the OpenAI GPT-3 model to use.
# #         prompt: The prompt template to query GPT-3 with.
# #         num_retries: Number of times to retry OpenAI call in the event of an API failure.
# #     Returns:
# #         edited_claim: The edited claim.
# #     """
# #     if context:
# #         gpt3_input = prompt.format(
# #             context=context, claim=claim, query=query, evidence=evidence
# #         ).strip()
# #     else:
# #         gpt3_input = prompt.format(claim=claim, query=query, evidence=evidence).strip()

# #     for _ in range(num_retries):
# #         try:
# #             response = openai.Completion.create(
# #                 model=model,
# #                 prompt=gpt3_input,
# #                 temperature=0.0,
# #                 max_tokens=512,
# #                 stop=["\n\n"],
# #             )
# #             break
# #         except openai.error.OpenAIError as exception:
# #             print(f"{exception}. Retrying...")
# #             time.sleep(2)

# #     edited_claim = parse_api_response(response.choices[0].text)
# #     # If there was an error in GPT-3 generation, return the claim.
# #     if not edited_claim:
# #         edited_claim = claim
# #     output = {"text": edited_claim}
# #     return output