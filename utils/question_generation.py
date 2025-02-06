"""
question_generation.py

Module for generating fact-checking questions for atomic statements in the RARR framework.
"""

import time
import logging
from typing import List, Dict, Optional
from openai.error import OpenAIError
from utils.newfile import call_openai_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# QUESTION PARSING & PROMPT GENERATION
# --------------------------------------------------------------------------------
def parse_api_response(api_response: str, required_questions: int = 4) -> List[str]:
    """
    Extract up to `required_questions` questions from the GPT response text.
    Ensures we return exactly `required_questions` questions.

    Args:
        api_response (str): Raw GPT response.
        required_questions (int): Number of required questions.

    Returns:
        List[str]: Cleaned and structured fact-checking questions.
    """
    try:
        questions = []
        for line in api_response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                split_index = line.find(" ")
                if split_index != -1:
                    question = line[split_index:].strip().lstrip(".- )")
                    if question:
                        if not question[0].isupper():
                            question = question[0].upper() + question[1:]
                        questions.append(question)

        unique_questions = list(dict.fromkeys(questions))

        if len(unique_questions) > required_questions:
            return unique_questions[:required_questions]

        while len(unique_questions) < required_questions:
            unique_questions.append(
                "What sources can verify this claim?"
            )

        return unique_questions

    except Exception as e:
        logger.error(f"Error parsing API response: {e}")
        return [
            "Who is involved in this claim?",
            "What are the key details supporting this claim?",
            "Why is this claim significant or relevant?",
            "When did this event or claim take place?"
        ][:required_questions]


def generate_contextual_prompt(
    claim: str,
    context: Optional[str] = None,
    previous_questions: Optional[List[str]] = None
) -> str:
    """
    Builds a user prompt for GPT, including context (previous statements)
    and previously asked questions to avoid repetition.
    """
    prompt_parts = []
    
    if context:
        prompt_parts.append(
            "Context (previous statements or text):\n"
            f"{context}\n\n"
            "Current statement to verify:\n"
            f"{claim}\n"
        )
    else:
        prompt_parts.append(f"Statement to verify:\n{claim}\n")
    
    prompt_parts.append(
        "\nGenerate exactly 4 specific fact-checking questions that:\n"
        "1. Identify key entities involved in the claim\n"
        "2. Extract crucial supporting details\n"
        "3. Determine the significance of the claim\n"
        "4. Establish the timeline of the event or statement\n"
    )
    
    if previous_questions:
        prompt_parts.append("\nPreviously asked questions (avoid repeating):")
        for q in previous_questions:
            prompt_parts.append(f"- {q}")
    
    prompt_parts.append("\nYour new questions (list them as 1., 2., 3., 4.):")
    return "\n".join(prompt_parts)


def run_rarr_question_generation(
    claim: str,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    num_retries: int = 5,
    context: Optional[str] = None,
    previous_questions: Optional[List[str]] = None,
    required_questions: int = 4
) -> List[str]:
    """
    Generates `required_questions` fact-checking questions for a given claim.
    - Leverages `context` (previous statements) and `previous_questions` to avoid duplicates.
    - Retries up to `num_retries` times on transient errors.
    - Ensures we end up with exactly `required_questions` unique questions.
    - `prompt` can be a base prompt that the user wants appended to the final GPT call.
    """
    try:
        contextual_prompt = generate_contextual_prompt(claim, context, previous_questions)
        system_prompt = (
            "You are a fact-checking assistant specialized in generating specific, "
            "focused questions to verify factual claims. Keep them concise and fact-driven."
        )
        gpt_input = f"{prompt}\n{contextual_prompt}".strip()

        all_questions = set()
        attempts = 0
        max_rounds = 3  

        while len(all_questions) < required_questions and attempts < max_rounds:
            for attempt_i in range(num_retries):
                try:
                    response = call_openai_api(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": gpt_input}
                        ],
                        model=model,
                        temperature=temperature,
                        max_retries=1,
                        use_cache=True
                    )

                    raw_text = response["choices"][0]["message"]["content"]
                    
                    new_questions = parse_api_response(raw_text, required_questions)

                    for q in new_questions:
                        all_questions.add(q)

                    break  

                except OpenAIError as oe:
                    logger.warning(f"OpenAI API Error (attempt {attempt_i + 1}/{num_retries}): {oe}")
                    if attempt_i < num_retries - 1:
                        time.sleep(2)
                except Exception as e:
                    logger.warning(f"Unexpected Error (attempt {attempt_i + 1}/{num_retries}): {e}")
                    if attempt_i < num_retries - 1:
                        time.sleep(2)

            attempts += 1

        questions_list = list(all_questions)[:required_questions]

        while len(questions_list) < required_questions:
            questions_list.append(
                f"What evidence supports the claim that {claim.lower().rstrip('.')}?"
            )

        return questions_list

    except Exception as e:
        logger.error(f"Error in question generation: {e}")
        return [
            "Who is involved in this claim?",
            "What are the key details supporting this claim?",
            "Why is this claim significant or relevant?",
            "When did this event or claim take place?"
        ][:required_questions]


def process_atomic_statements(
    statements: List[str],
    model: str,
    base_prompt: str,
    temperature: float = 0.7
) -> Dict[str, List[str]]:
    """
    Processes multiple statements to generate fact-checking questions.

    Args:
        statements (List[str]): List of atomic statements.
        model (str): OpenAI model to use.
        base_prompt (str): Instruction for question generation.
        temperature (float): Sampling temperature.

    Returns:
        Dict[str, List[str]]: Mapping of statements to generated questions.
    """
    try:
        results = {}
        
        for i, statement in enumerate(statements):
            logger.info(f"Processing statement {i + 1}/{len(statements)}")

            questions = run_rarr_question_generation(
                claim=statement,
                model=model,
                prompt=base_prompt,
                temperature=temperature,
                num_retries=3,
                required_questions=4
            )

            results[statement] = questions
            logger.info(f"Generated {len(questions)} questions for statement {i + 1}")

        return results

    except Exception as e:
        logger.error(f"Error processing atomic statements: {e}")
        fallback = [
            "Who is involved in this claim?",
            "What are the key details supporting this claim?",
            "Why is this claim significant or relevant?",
            "When did this event or claim take place?"
        ]
        return {statement: fallback for statement in statements}


if __name__ == "__main__":
    example_statements = [
        "Lena Headey portrayed Cersei Lannister in HBO's Game of Thrones since 2011.",
        "She received three consecutive Emmy nominations for the role.",
        "She became one of the highest-paid television actors by 2017."
    ]

    base_prompt = "Generate fact-checking questions to verify the following information."

    results = process_atomic_statements(
        statements=example_statements,
        model="gpt-3.5-turbo",
        base_prompt=base_prompt,
        temperature=0.7
    )

    for statement, questions in results.items():
        print(f"\nStatement: {statement}")
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")

# """
# question_generation.py

# Module for generating fact-checking questions for atomic statements in the RARR framework.
# """

# import time
# import logging
# from typing import List, Dict, Optional
# from openai.error import OpenAIError
# from utils.newfile import call_openai_api

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
# logger = logging.getLogger(__name__)

# # --------------------------------------------------------------------------------
# # QUESTION PARSING & PROMPT GENERATION
# # --------------------------------------------------------------------------------
# def parse_api_response(api_response: str, required_questions: int = 3) -> List[str]:
#     """
#     Extracts up to `required_questions` questions from the GPT response text.
#     Ensures we return exactly `required_questions` questions.

#     Args:
#         api_response (str): Raw GPT response.
#         required_questions (int): Number of required questions.

#     Returns:
#         List[str]: Cleaned and structured fact-checking questions.
#     """
#     try:
#         questions = []
#         for line in api_response.split("\n"):
#             line = line.strip()
#             if line and line[0].isdigit():
#                 split_index = line.find(" ")
#                 if split_index != -1:
#                     question = line[split_index:].strip().lstrip(".- )")
#                     if question:
#                         if not question[0].isupper():
#                             question = question[0].upper() + question[1:]
#                         questions.append(question)

#         unique_questions = list(dict.fromkeys(questions))

#         if len(unique_questions) > required_questions:
#             return unique_questions[:required_questions]

#         while len(unique_questions) < required_questions:
#             if not unique_questions:
#                 unique_questions.append("What specific evidence supports this statement?")
#             else:
#                 base_q = unique_questions[-1].rstrip("?")
#                 new_q = f"What sources or evidence can verify that {base_q.lower()}?"
#                 unique_questions.append(new_q)

#         return unique_questions

#     except Exception as e:
#         logger.error(f"Error parsing API response: {e}")
#         return [
#             "What evidence supports this statement?",
#             "What sources can verify this claim?",
#             "How can this information be validated?"
#         ][:required_questions]


# def generate_prompt(
#     claim: str,
#     base_prompt: str,
#     context: Optional[str] = None,
#     previous_questions: Optional[List[str]] = None
# ) -> str:
#     """
#     Generates a structured prompt for GPT with necessary context.

#     Args:
#         claim (str): Statement to verify.
#         base_prompt (str): The main instruction for GPT.
#         context (Optional[str]): Previous context (if any).
#         previous_questions (Optional[List[str]]): Questions asked before.

#     Returns:
#         str: The final structured prompt.
#     """
#     prompt_parts = [f"{base_prompt}\n\nStatement to verify:\n{claim}"]

#     if context:
#         prompt_parts.append(f"\nContext (previous statements or text):\n{context}")

#     prompt_parts.append(
#         "\nGenerate exactly 3 specific questions that:\n"
#         "1. Focus on verifiable facts and claims\n"
#         "2. Target specific details that can be fact-checked\n"
#         "3. Can be answered with credible sources\n"
#     )

#     if previous_questions:
#         prompt_parts.append("\nPreviously asked questions (avoid repeating):")
#         for q in previous_questions:
#             prompt_parts.append(f"- {q}")

#     prompt_parts.append("\nYour new questions (list them as 1., 2., 3.):")

#     return "\n".join(prompt_parts)


# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     base_prompt: str,
#     temperature: float = 0.7,
#     num_retries: int = 3,
#     context: Optional[str] = None,
#     previous_questions: Optional[List[str]] = None,
#     required_questions: int = 3
# ) -> List[str]:
#     """
#     Generates fact-checking questions using GPT.

#     Args:
#         claim (str): The statement for verification.
#         model (str): OpenAI model to use.
#         base_prompt (str): The main instruction for GPT.
#         temperature (float): Sampling temperature.
#         num_retries (int): Number of retries in case of failure.
#         context (Optional[str]): Previous statements as context.
#         previous_questions (Optional[List[str]]): Previously asked questions.
#         required_questions (int): Number of required questions.

#     Returns:
#         List[str]: List of fact-checking questions.
#     """
#     try:
#         all_questions = set()
#         attempts = 0
#         max_rounds = 3  

#         while len(all_questions) < required_questions and attempts < max_rounds:
#             for attempt_i in range(num_retries):
#                 try:
#                     prompt = generate_prompt(
#                         claim=claim, 
#                         base_prompt=base_prompt,
#                         context=context, 
#                         previous_questions=previous_questions
#                     )

#                     response = call_openai_api(
#                         messages=[
#                             {"role": "system", "content": "You are a fact-checking assistant."},
#                             {"role": "user", "content": prompt}
#                         ],
#                         model=model,
#                         temperature=temperature,
#                         max_retries=1,
#                         use_cache=True
#                     )

#                     raw_text = response["choices"][0]["message"]["content"]
#                     new_questions = parse_api_response(raw_text, required_questions)

#                     for q in new_questions:
#                         if q not in all_questions:
#                             all_questions.add(q)

#                     break  

#                 except OpenAIError as oe:
#                     logger.warning(f"OpenAI API Error (attempt {attempt_i + 1}/{num_retries}): {oe}")
#                     if attempt_i < num_retries - 1:
#                         time.sleep(2)
#                 except Exception as e:
#                     logger.warning(f"Unexpected Error (attempt {attempt_i + 1}/{num_retries}): {e}")
#                     if attempt_i < num_retries - 1:
#                         time.sleep(2)

#             attempts += 1

#         questions_list = list(all_questions)[:required_questions]

#         while len(questions_list) < required_questions:
#             questions_list.append(
#                 f"What evidence supports the claim that {claim.lower().rstrip('.')}?"
#             )

#         return questions_list

#     except Exception as e:
#         logger.error(f"Error in question generation: {e}")
#         return [
#             f"What evidence supports the claim that {claim.lower().rstrip('.')}?",
#             "What sources can verify this information?",
#             "How can these specific details be validated?"
#         ][:required_questions]


# def process_atomic_statements(
#     statements: List[str],
#     model: str,
#     base_prompt: str,
#     temperature: float = 0.7
# ) -> Dict[str, List[str]]:
#     """
#     Processes multiple statements to generate fact-checking questions.

#     Args:
#         statements (List[str]): List of atomic statements.
#         model (str): OpenAI model to use.
#         base_prompt (str): Instruction for question generation.
#         temperature (float): Sampling temperature.

#     Returns:
#         Dict[str, List[str]]: Mapping of statements to generated questions.
#     """
#     try:
#         results = {}
#         all_previous_questions: List[str] = []

#         for i, statement in enumerate(statements):
#             logger.info(f"Processing statement {i + 1}/{len(statements)}")

#             context = " ".join(statements[:i]) if i > 0 else ""

#             questions = run_rarr_question_generation(
#                 claim=statement,
#                 model=model,
#                 base_prompt=base_prompt,
#                 temperature=temperature,
#                 num_retries=3,
#                 context=context,
#                 previous_questions=all_previous_questions,
#                 required_questions=3
#             )

#             results[statement] = questions
#             all_previous_questions.extend(questions)
#             logger.info(f"Generated {len(questions)} questions for statement {i + 1}")

#         return results

#     except Exception as e:
#         logger.error(f"Error processing atomic statements: {e}")
#         return {statement: ["What evidence supports this statement?", "What sources can verify this information?", "How can these claims be validated?"] for statement in statements}


# if __name__ == "__main__":
#     example_statements = [
#         "Lena Headey portrayed Cersei Lannister in HBO's Game of Thrones since 2011.",
#         "She received three consecutive Emmy nominations for the role.",
#         "She became one of the highest-paid television actors by 2017."
#     ]

#     base_prompt = "Generate fact-checking questions to verify the following information."

#     results = process_atomic_statements(
#         statements=example_statements,
#         model="gpt-3.5-turbo",
#         base_prompt=base_prompt,
#         temperature=0.7
#     )

#     for statement, questions in results.items():
#         print(f"\nStatement: {statement}")
#         for i, question in enumerate(questions, 1):
#             print(f"{i}. {question}")



# """
# question_generation.py

# Module for generating fact-checking questions for atomic statements in the RARR framework.
# """

# import time
# import logging
# from typing import List, Dict, Optional

# import openai
# from openai.error import OpenAIError  # <-- Add this line


# from utils.newfile import call_openai_api


# # Import your custom OpenAI wrapper from utils
# from newfile import call_openai_api

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# def parse_api_response(api_response: str, required_questions: int = 3) -> List[str]:
#     """
#     Extract up to `required_questions` questions from the GPT response text.
#     Ensures we return exactly `required_questions` questions,
#     padding with generic placeholders if fewer are found.
#     """
#     try:
#         questions = []
#         # Split on new lines, looking for lines starting with a digit (e.g., "1.", "2)").
#         for line in api_response.split("\n"):
#             line = line.strip()
#             if line and line[0].isdigit():
#                 # Find the first space after the digit or punctuation
#                 split_index = line.find(" ")
#                 if split_index != -1:
#                     # Clean up leftover punctuation
#                     question = line[split_index:].strip().lstrip(".- )")
#                     if question:
#                         # Ensure question starts uppercase for consistency
#                         if not question[0].isupper():
#                             question = question[0].upper() + question[1:]
#                         questions.append(question)

#         # Remove duplicates (preserving first occurrence)
#         unique_questions = list(dict.fromkeys(questions))

#         # If we got more than needed, truncate
#         if len(unique_questions) > required_questions:
#             return unique_questions[:required_questions]

#         # If fewer, pad with fallback questions
#         while len(unique_questions) < required_questions:
#             if not unique_questions:
#                 unique_questions.append("What specific evidence supports this statement?")
#             else:
#                 base_q = unique_questions[-1].rstrip("?")
#                 new_q = f"What sources or evidence can verify that {base_q.lower()}?"
#                 unique_questions.append(new_q)

#         return unique_questions

#     except Exception as e:
#         logger.error(f"Error parsing API response: {e}")
#         # Fallback if parsing fails
#         return [
#             "Who is involved in this claim?",
#             "What are the key details supporting this claim?",
#             "Why is this claim significant or relevant?",
#             "When did this event or claim take place?"
#         ][:required_questions]


# def generate_contextual_prompt(
#     claim: str,
#     context: Optional[str] = None,
#     previous_questions: Optional[List[str]] = None
# ) -> str:
#     """
#     Builds a user prompt for GPT, including context (previous statements)
#     and previously asked questions to avoid repetition.
#     """
#     prompt_parts = []
    
#     if context:
#         prompt_parts.append(
#             "Context (previous statements or text):\n"
#             f"{context}\n\n"
#             "Current statement to verify:\n"
#             f"{claim}\n"
#         )
#     else:
#         prompt_parts.append(f"Statement to verify:\n{claim}\n")
    
#     prompt_parts.append(
#         "\nGenerate exactly 3 specific questions that:\n"
#         "1. Focus on verifiable facts and claims\n"
#         "2. Target specific details that can be fact-checked\n"
#         "3. Can be answered with credible sources\n"
#     )
    
#     if previous_questions:
#         prompt_parts.append("\nPreviously asked questions (avoid repeating):")
#         for q in previous_questions:
#             prompt_parts.append(f"- {q}")
    
#     prompt_parts.append("\nYour new questions (list them as 1., 2., 3.):")
#     return "\n".join(prompt_parts)


# def are_questions_similar(q1: str, q2: str, similarity_threshold: float = 0.8) -> bool:
#     """
#     Simple word-overlap-based similarity check to prevent repeated or near-duplicate questions.
#     If overlap is above `similarity_threshold`, they're considered similar.
#     """
#     try:
#         common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
#         words1 = {w.lower() for w in q1.split() if w.lower() not in common_words}
#         words2 = {w.lower() for w in q2.split() if w.lower() not in common_words}
        
#         if not words1 or not words2:
#             return False

#         overlap = len(words1.intersection(words2))
#         similarity = overlap / max(len(words1), len(words2))
#         return similarity > similarity_threshold

#     except Exception as e:
#         logger.error(f"Error comparing questions for similarity: {e}")
#         return False


# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     prompt: str,
#     temperature: float = 0.7,
#     num_retries: int = 5,
#     context: Optional[str] = None,
#     previous_questions: Optional[List[str]] = None,
#     required_questions: int = 4
# ) -> List[str]:
#     """
#     Generates `required_questions` fact-checking questions for a given claim.
#     - Leverages `context` (previous statements) and `previous_questions` to avoid duplicates.
#     - Retries up to `num_retries` times on transient errors.
#     - Ensures we end up with exactly `required_questions` unique questions.
#     - `prompt` can be a base prompt that the user wants appended to the final GPT call.
#     """
#     try:
#         # Build the final user prompt (user content)
#         contextual_prompt = generate_contextual_prompt(claim, context, previous_questions)
#         system_prompt = (
#             "You are a fact-checking assistant specialized in generating specific, "
#             "focused questions to verify factual claims. Keep it short and on-topic."
#         )
#         gpt_input = f"{prompt}\n{contextual_prompt}".strip()

#         all_questions = set()
#         attempts = 0
#         max_rounds = 3  # The number of GPT call rounds to gather more questions if needed

#         while len(all_questions) < required_questions and attempts < max_rounds:
#             for attempt_i in range(num_retries):
#                 try:
#                     # Use call_openai_api for caching & retry logic
#                     response = call_openai_api(
#                         messages=[
#                             {"role": "system", "content": system_prompt},
#                             {"role": "user", "content": gpt_input}
#                         ],
#                         model=model,
#                         temperature=temperature,
#                         max_retries=1,  # Let the outer loop handle multiple attempts
#                         use_cache=True
#                     )

#                     raw_text = response["choices"][0]["message"]["content"]
                    
#                     # Parse the GPT output into question strings
#                     new_questions = parse_api_response(
#                         api_response=raw_text,
#                         required_questions=required_questions
#                     )
                    
#                     # Filter out questions too similar to what's already generated
#                     for q in new_questions:
#                         if not any(are_questions_similar(q, existing_q) for existing_q in all_questions):
#                             all_questions.add(q)

#                     break  # Successfully retrieved questions -> break the retry loop

#                 except OpenAIError as oe:
#                     logger.warning(f"OpenAI API Error (attempt {attempt_i + 1}/{num_retries}): {oe}")
#                     if attempt_i < num_retries - 1:
#                         time.sleep(2)  # short backoff
#                 except Exception as e:
#                     logger.warning(f"Unexpected Error (attempt {attempt_i + 1}/{num_retries}): {e}")
#                     if attempt_i < num_retries - 1:
#                         time.sleep(2)  # short backoff

#             attempts += 1

#         # Convert the set to a list, limit to required_questions
#         questions_list = list(all_questions)[:required_questions]

#         # If still fewer, fill with placeholders
#         while len(questions_list) < required_questions:
#             questions_list.append(
#                 f"What evidence supports the claim that {claim.lower().rstrip('.')}?"
#             )

#         return questions_list

#     except Exception as e:
#         logger.error(f"Error in question generation: {e}")
#         # Fallback if all fails
#         return [
#             f"What evidence supports the claim that {claim.lower().rstrip('.')}?",
#             "What sources can verify this information?",
#             "How can these specific details be validated?"
#         ][:required_questions]


# def process_atomic_statements(
#     statements: List[str],
#     model: str,
#     base_prompt: str,
#     temperature: float = 0.7
# ) -> Dict[str, List[str]]:
#     """
#     For each statement in `statements`, generates `required_questions` (default 3) questions
#     using `run_rarr_question_generation`. Maintains a set of previously asked questions across
#     statements to reduce duplicates. Returns a dict: {statement: [questions]}.
#     """
#     try:
#         results = {}
#         all_previous_questions: List[str] = []
        
#         for i, statement in enumerate(statements):
#             logger.info(f"Processing statement {i + 1}/{len(statements)}")

#             # For context, we pass all previous statements (unrevised),
#             # or you could pass only the last N if you want shorter prompts
#             context = " ".join(statements[:i]) if i > 0 else ""

#             questions = run_rarr_question_generation(
#                 claim=statement,
#                 model=model,
#                 prompt=base_prompt,
#                 temperature=temperature,
#                 num_retries=3,
#                 context=context,
#                 previous_questions=all_previous_questions,
#                 required_questions=3
#             )
            
#             results[statement] = questions
#             # Add newly generated questions to the global list to avoid duplication
#             all_previous_questions.extend(questions)
#             logger.info(f"Generated {len(questions)} questions for statement {i + 1}")
        
#         return results

#     except Exception as e:
#         logger.error(f"Error processing atomic statements: {e}")
#         fallback = [
#             "What evidence supports this statement?",
#             "What sources can verify this information?",
#             "How can these claims be validated?"
#         ]
#         return {statement: fallback for statement in statements}


# if __name__ == "__main__":
#     # Simple usage example
#     example_statements = [
#         "Lena Headey portrayed Cersei Lannister in HBO's Game of Thrones since 2011.",
#         "She received three consecutive Emmy nominations for the role.",
#         "She became one of the highest-paid television actors by 2017."
#     ]
    
#     example_prompt = "Generate specific questions to verify the following information."
    
#     results = process_atomic_statements(
#         statements=example_statements,
#         model="gpt-3.5-turbo",
#         base_prompt=example_prompt,
#         temperature=0.7
#     )
    
#     print("\nGenerated Questions per Statement:")
#     for statement, questions in results.items():
#         print(f"\nStatement: {statement}")
#         for i, question in enumerate(questions, 1):
#             print(f"{i}. {question}")



# """
# question_generation.py

# Module for generating fact-checking questions for atomic statements in the RARR framework.
# """

# import os
# import time
# import logging
# from typing import List, Dict, Optional

# import openai
# from openai.error import OpenAIError

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")


# def parse_api_response(api_response: str, required_questions: int = 3) -> List[str]:
#     """
#     Extract up to `required_questions` questions from the GPT response text.
#     Ensures we return exactly `required_questions` questions,
#     padding with generic placeholders if fewer are found.
#     """
#     try:
#         questions = []
#         # Split on new lines, looking for lines starting with a digit (e.g., "1.", "2)").
#         for line in api_response.split("\n"):
#             line = line.strip()
#             if line and line[0].isdigit():
#                 # Find the first space after the digit or punctuation
#                 split_index = line.find(" ")
#                 if split_index != -1:
#                     # Clean up leftover punctuation
#                     question = line[split_index:].strip().lstrip(".- )")
#                     # Ensure question starts uppercase for consistency
#                     if question and question[0].isupper():
#                         questions.append(question)

#         # Remove duplicates (preserving first occurrence)
#         unique_questions = list(dict.fromkeys(questions))

#         # If we got more than needed, truncate
#         if len(unique_questions) > required_questions:
#             return unique_questions[:required_questions]

#         # If fewer, pad with fallback questions
#         while len(unique_questions) < required_questions:
#             if not unique_questions:
#                 unique_questions.append("What specific evidence supports this statement?")
#             else:
#                 base_q = unique_questions[-1].rstrip("?")
#                 new_q = f"What sources or evidence can verify that {base_q.lower()}?"
#                 unique_questions.append(new_q)
        
#         return unique_questions

#     except Exception as e:
#         logger.error(f"Error parsing API response: {e}")
#         # Fallback if parsing fails
#         return [
#             "What evidence supports this statement?",        
#             "What sources can verify this claim?",
#             "How can this information be validated?"
#         ][:required_questions]


# def generate_contextual_prompt(
#     claim: str,
#     context: Optional[str] = None,
#     previous_questions: Optional[List[str]] = None
# ) -> str:
#     """
#     Builds a user prompt for GPT, including context (previous statements)
#     and previously asked questions to avoid repetition.
#     """
#     prompt_parts = []
    
#     if context:
#         prompt_parts.append(
#             "Previous context for reference:\n"
#             f"{context}\n\n"
#             "Current statement to verify:\n"
#             f"{claim}\n"
#         )
#     else:
#         prompt_parts.append(f"Statement to verify:\n{claim}\n")
    
#     prompt_parts.append(
#         "\nGenerate exactly 3 specific questions that:"
#         "\n1. Focus on verifiable facts and claims"
#         "\n2. Target specific details that can be fact-checked"
#         "\n3. Can be answered with credible sources"
#     )
    
#     if previous_questions:
#         prompt_parts.append("\nPreviously asked questions (avoid repeating):")
#         for q in previous_questions:
#             prompt_parts.append(f"- {q}")
    
#     prompt_parts.append("\nNew questions:")
#     return "\n".join(prompt_parts)


# def are_questions_similar(q1: str, q2: str, similarity_threshold: float = 0.8) -> bool:
#     """
#     Simple word-overlap-based similarity check to prevent repeated questions.
#     If overlap is above `similarity_threshold`, they're considered similar.
#     """
#     try:
#         common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
#         words1 = {w.lower() for w in q1.split() if w.lower() not in common_words}
#         words2 = {w.lower() for w in q2.split() if w.lower() not in common_words}
        
#         if not words1 or not words2:
#             return False
            
#         overlap = len(words1.intersection(words2))
#         similarity = overlap / max(len(words1), len(words2))
        
#         return similarity > similarity_threshold

#     except Exception as e:
#         logger.error(f"Error comparing questions: {e}")
#         return False


# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     prompt: str,
#     temperature: float = 0.7,
#     num_retries: int = 5,
#     context: Optional[str] = None,
#     previous_questions: Optional[List[str]] = None,
#     required_questions: int = 3
# ) -> List[str]:
#     """
#     Generates `required_questions` fact-checking questions for a given claim.
#     - Leverages `context` (previous statements) and `previous_questions` to avoid duplicates.
#     - Retries up to `num_retries` times on transient errors.
#     - Ensures we end up with exactly `required_questions` unique questions.
#     """
#     try:
#         # Build the final user prompt
#         contextual_prompt = generate_contextual_prompt(claim, context, previous_questions)
#         system_prompt = (
#             "You are a fact-checking assistant specialized in generating specific, "
#             "focused questions to verify factual claims."
#         )
#         gpt_input = f"{prompt}\n{contextual_prompt}".strip()

#         all_questions = set()
#         attempts = 0
#         max_rounds = 3  # The number of GPT call rounds

#         # Attempt multiple GPT calls, stopping early if we reach the required number of questions
#         while len(all_questions) < required_questions and attempts < max_rounds:
#             for attempt_i in range(num_retries):
#                 try:
#                     # Create a ChatCompletion with the 'system' + 'user' message
#                     response = openai.ChatCompletion.create(
#                         model=model,
#                         messages=[
#                             {"role": "system", "content": system_prompt},
#                             {"role": "user", "content": gpt_input}
#                         ],
#                         temperature=temperature
#                     )

#                     # Parse the GPT output
#                     new_questions = parse_api_response(
#                         response.choices[0].message.content,
#                         required_questions
#                     )
                    
#                     # Filter out questions too similar to what's already generated
#                     for q in new_questions:
#                         if not any(are_questions_similar(q, existing_q) for existing_q in all_questions):
#                             all_questions.add(q)
#                     break  # Successfully retrieved questions, break the retry loop

#                 except OpenAIError as oe:
#                     logger.warning(f"OpenAI API Error (attempt {attempt_i + 1}/{num_retries}): {oe}")
#                     if attempt_i < num_retries - 1:
#                         time.sleep(2)  # short backoff before next attempt
#                 except Exception as e:
#                     logger.warning(f"API Error (attempt {attempt_i + 1}/{num_retries}): {e}")
#                     if attempt_i < num_retries - 1:
#                         time.sleep(2)  # short backoff before next attempt

#             attempts += 1

#         # Limit to required_questions
#         questions_list = list(all_questions)[:required_questions]

#         # If still fewer, fill with placeholders
#         while len(questions_list) < required_questions:
#             questions_list.append(
#                 f"What evidence supports the claim that {claim.lower().rstrip('.')}?"
#             )

#         return questions_list

#     except Exception as e:
#         logger.error(f"Error in question generation: {e}")
#         # Fallback if all fails
#         return [
#             f"What evidence supports the claim that {claim.lower().rstrip('.')}?",
#             "What sources can verify this information?",
#             "How can these specific details be validated?"
#         ][:required_questions]


# def process_atomic_statements(
#     statements: List[str],
#     model: str,
#     base_prompt: str,
#     temperature: float = 0.7
# ) -> Dict[str, List[str]]:
#     """
#     For each statement in `statements`, generates 3 questions using `run_rarr_question_generation`.
#     Maintains a set of previously asked questions across statements to reduce duplicates.
#     Returns a dict: {statement: [questions]}
#     """
#     try:
#         results = {}
#         # Accumulate all previously generated questions to avoid repeats
#         all_previous_questions = []
        
#         for i, statement in enumerate(statements):
#             logger.info(f"Processing statement {i + 1}/{len(statements)}")

#             # For context, we can pass all prior statements (unrevised)
#             context = " ".join(statements[:i]) if i > 0 else ""

#             questions = run_rarr_question_generation(
#                 claim=statement,
#                 model=model,
#                 prompt=base_prompt,
#                 temperature=temperature,
#                 num_retries=3,
#                 context=context,
#                 previous_questions=all_previous_questions,
#                 required_questions=3
#             )
            
#             results[statement] = questions
#             # Add newly generated questions to the global list to avoid duplication later
#             all_previous_questions.extend(questions)
#             logger.info(f"Generated {len(questions)} questions for statement {i + 1}")
        
#         return results

#     except Exception as e:
#         logger.error(f"Error processing atomic statements: {e}")
#         fallback = [
#             "What evidence supports this statement?",
#             "What sources can verify this information?",
#             "How can these claims be validated?"
#         ]
#         return {statement: fallback for statement in statements}


# # Example direct usage:
# if __name__ == "__main__":
#     # Example usage
#     example_statements = [
#         "Lena Headey portrayed Cersei Lannister in HBO's Game of Thrones since 2011.",
#         "She received three consecutive Emmy nominations for the role.",
#         "She became one of the highest-paid television actors by 2017."
#     ]
    
#     example_prompt = "Generate specific questions to verify the following information."
    
#     results = process_atomic_statements(
#         statements=example_statements,
#         model="gpt-3.5-turbo",
#         base_prompt=example_prompt,
#         temperature=0.7
#     )
    
#     print("\nGenerated Questions per Statement:")
#     for statement, questions in results.items():
#         print(f"\nStatement: {statement}")
#         for i, question in enumerate(questions, 1):
#             print(f"{i}. {question}")


# """Utils for running question generation."""
# import os
# import time
# from typing import List
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
#         # Match lines that start with a number (e.g., "1. ", "2)")
#         if line and line[0].isdigit():
#             # Extract question text after the number and optional punctuation
#             split_index = line.find(" ")  # Find the first space
#             if split_index != -1:
#                 question = line[split_index + 1:].strip()
#                 questions.append(question)
#     return questions


# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     prompt: str,
#     temperature: float = 0.7,
#     num_rounds: int = 1,
#     context: str = None,
#     num_retries: int = 5,
# ) -> List[str]:
#     """Generates questions that interrogate the information in a claim.

#     Args:
#         claim: Text to generate questions for.
#         model: Name of the OpenAI GPT model to use.
#         prompt: The prompt template to query GPT with.
#         temperature: Temperature for GPT sampling. 0 for deterministic output.
#         num_rounds: Number of times to sample questions for diversity.
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
#         for attempt in range(num_retries):
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
#                 # Parse response for questions
#                 cur_round_questions = parse_api_response(response.choices[0].message["content"].strip())
#                 questions.update(cur_round_questions)
#                 break  # Exit retry loop on success
#             except openai.error.OpenAIError as exception:
#                 print(f"API Error: {exception}. Retrying... ({attempt + 1}/{num_retries})")
#                 time.sleep(1)  # Wait before retrying
#         else:
#             print(f"Failed to generate questions after {num_retries} retries.")

#     return list(sorted(questions))


# if __name__ == "__main__":
#     # Example usage (for testing purposes)
#     example_claim = "Albert Einstein developed the theory of relativity."
#     example_prompt = """You said: {claim}
#     To verify, I will ask questions:
#     1. Who developed the theory of relativity?
#     2. What is the theory of relativity about?"""

#     questions = run_rarr_question_generation(
#         claim=example_claim,
#         model="gpt-3.5-turbo",
#         prompt=example_prompt,
#         temperature=0.7,
#         num_rounds=3,
#     )
#     print("Generated Questions:")
#     for question in questions:
#         print(f"- {question}")






# """Utils for running question generation."""
# import os
# import time
# from typing import List

# import openai

# # Set up OpenAI API key from environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def parse_api_response(api_response: str) -> List[str]:
#     """Extracts questions from the model's response.

#     The prompt returns questions as a string in an ordered list format.
#     This function parses the response and extracts individual questions.

#     Args:
#         api_response: Question generation response from the model.
#     Returns:
#         questions: A list of parsed questions.
#     """
#     search_string = "I googled:"
#     questions = []
#     for question in api_response.split("\n"):
#         # Only consider lines that contain the search string
#         if search_string in question:
#             question = question.split(search_string)[1].strip()
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

#     Given a claim, this function uses the model to generate questions that test or
#     clarify the information within the claim. It samples questions multiple times 
#     to achieve diversity.

#     Args:
#         claim: Text from which to generate questions.
#         model: Name of the OpenAI model to use (e.g., "gpt-3.5-turbo" or "text-davinci-003").
#         prompt: The prompt template to query the model with.
#         temperature: Temperature for sampling. 0 represents greedy decoding.
#         num_rounds: Number of times to sample questions.
#         context: Optional context to include with the claim.
#         num_retries: Number of retries in case of API failures.
#     Returns:
#         questions: A list of generated questions.
#     """
#     # Format the input prompt with optional context
#     if context:
#         formatted_input = prompt.format(context=context, claim=claim).strip()
#     else:
#         formatted_input = prompt.format(claim=claim).strip()

#     questions = set()
#     for _ in range(num_rounds):
#         for _ in range(num_retries):
#             try:
#                 # Check if the model is a chat model (e.g., gpt-3.5-turbo)
#                 if "gpt-3.5-turbo" in model or "gpt-4" in model:
#                     response = openai.ChatCompletion.create(
#                         model=model,
#                         messages=[{"role": "user", "content": formatted_input}],
#                         temperature=temperature,
#                         max_tokens=256,
#                     )
#                     response_text = response.choices[0].message["content"].strip()
#                 else:
#                     # Use regular Completion endpoint for non-chat models
#                     response = openai.Completion.create(
#                         model=model,
#                         prompt=formatted_input,
#                         temperature=temperature,
#                         max_tokens=256,
#                     )
#                     response_text = response.choices[0].text.strip()

#                 # Parse and add questions from the current response
#                 cur_round_questions = parse_api_response(response_text)
#                 questions.update(cur_round_questions)
#                 break

#             except openai.error.OpenAIError as exception:
#                 print(f"{exception}. Retrying...")
#                 time.sleep(1)

#     # Return the sorted list of unique questions
#     return list(sorted(questions))




# """Utils for running question generation with context-aware atomic statement processing."""
# import os
# import time
# import logging
# from typing import List, Dict, Tuple, Optional

# import openai

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# openai.api_key = os.getenv("OPENAI_API_KEY")

# def parse_api_response(api_response: str, required_questions: int = 3) -> List[str]:
#     """Extract questions from the GPT API response."""
#     try:
#         questions = []
#         for line in api_response.split("\n"):
#             line = line.strip()
#             if line and line[0].isdigit():
#                 # Find first space after the number
#                 split_index = line.find(" ")
#                 if split_index != -1:
#                     # Extract question and clean it
#                     question = line[split_index:].strip()
#                     # Remove any leading punctuation or spaces
#                     question = question.lstrip(".- )")
#                     if question and question[0].isupper():  # Ensure it starts with capital letter
#                         questions.append(question)

#         # Ensure exactly required_questions unique questions
#         unique_questions = list(dict.fromkeys(questions))  # Remove duplicates
        
#         if len(unique_questions) > required_questions:
#             return unique_questions[:required_questions]
        
#         while len(unique_questions) < required_questions:
#             if not unique_questions:
#                 unique_questions.append("What specific evidence supports this statement?")
#             else:
#                 # Generate a variation of the last question
#                 base_q = unique_questions[-1].rstrip("?")
#                 new_q = f"What sources or evidence can verify that {base_q.lower()}?"
#                 unique_questions.append(new_q)
        
#         return unique_questions

#     except Exception as e:
#         logger.error(f"Error parsing API response: {e}")
#         # Fallback questions
#         return [
#             "What evidence supports this statement?",
#             "What sources can verify this claim?",
#             "How can this information be validated?"
#         ][:required_questions]

# def generate_contextual_prompt(
#     claim: str,
#     context: Optional[str] = None,
#     previous_questions: Optional[List[str]] = None
# ) -> str:
#     """Generate a fact-checking focused prompt with context awareness."""
#     prompt_parts = []
    
#     if context:
#         prompt_parts.append(
#             "Previous context for reference:\n"
#             f"{context}\n\n"
#             "Current statement to verify:\n"
#             f"{claim}\n"
#         )
#     else:
#         prompt_parts.append(f"Statement to verify:\n{claim}\n")
    
#     prompt_parts.append(
#         "\nGenerate exactly 3 specific questions that:"
#         "\n1. Focus on verifiable facts and claims"
#         "\n2. Target specific details that can be fact-checked"
#         "\n3. Can be answered with credible sources"
#     )
    
#     if previous_questions:
#         prompt_parts.append("\nPreviously asked questions (avoid repeating):")
#         for q in previous_questions:
#             prompt_parts.append(f"- {q}")
    
#     prompt_parts.append("\nNew questions:")
    
#     return "\n".join(prompt_parts)

# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     prompt: str,
#     temperature: float = 0.7,
#     num_retries: int = 5,
#     context: Optional[str] = None,
#     previous_questions: Optional[List[str]] = None,
#     required_questions: int = 3
# ) -> List[str]:
#     """Generates fact-checking questions with context awareness."""
#     try:
#         # Prepare prompt
#         contextual_prompt = generate_contextual_prompt(claim, context, previous_questions)
#         system_prompt = (
#             "You are a fact-checking assistant specialized in generating specific, "
#             "focused questions to verify factual claims. Focus on questions that can "
#             "be answered with concrete evidence and credible sources."
#         )
#         gpt_input = f"{prompt}\n{contextual_prompt}".strip()

#         questions = set()
#         attempts = 0
#         max_attempts = 3

#         while len(questions) < required_questions and attempts < max_attempts:
#             for attempt in range(num_retries):
#                 try:
#                     response = openai.ChatCompletion.create(
#                         model=model,
#                         messages=[
#                             {"role": "system", "content": system_prompt},
#                             {"role": "user", "content": gpt_input},
#                         ],
#                         temperature=temperature,
#                         max_tokens=256,
#                     )
                    
#                     new_questions = parse_api_response(
#                         response.choices[0].message["content"].strip(),
#                         required_questions
#                     )
                    
#                     # Filter out similar questions
#                     for q in new_questions:
#                         if not any(are_questions_similar(q, existing_q) 
#                                  for existing_q in questions):
#                             questions.add(q)
#                     break

#                 except openai.error.OpenAIError as e:
#                     logger.warning(f"API Error (attempt {attempt + 1}/{num_retries}): {e}")
#                     if attempt < num_retries - 1:
#                         time.sleep(2)

#             attempts += 1

#         questions_list = list(questions)[:required_questions]
#         while len(questions_list) < required_questions:
#             questions_list.append(
#                 f"What evidence supports the claim that {claim.lower().rstrip('.')}?"
#             )

#         return questions_list

#     except Exception as e:
#         logger.error(f"Error in question generation: {e}")
#         return [
#             f"What evidence supports the claim that {claim.lower().rstrip('.')}?",
#             "What sources can verify this information?",
#             "How can these specific details be validated?"
#         ][:required_questions]

# def are_questions_similar(q1: str, q2: str, similarity_threshold: float = 0.8) -> bool:
#     """Check for question similarity using word overlap."""
#     try:
#         # Convert to sets of words, ignoring common words
#         common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
#         words1 = {w.lower() for w in q1.split() if w.lower() not in common_words}
#         words2 = {w.lower() for w in q2.split() if w.lower() not in common_words}
        
#         if not words1 or not words2:
#             return False
            
#         overlap = len(words1.intersection(words2))
#         similarity = overlap / max(len(words1), len(words2))
        
#         return similarity > similarity_threshold

#     except Exception as e:
#         logger.error(f"Error comparing questions: {e}")
#         return False

# def process_atomic_statements(
#     statements: List[str],
#     model: str,
#     base_prompt: str,
#     temperature: float = 0.7
# ) -> Dict[str, List[str]]:
#     """Process atomic statements with context-aware question generation."""
#     try:
#         results = {}
#         all_previous_questions = []
        
#         for i, statement in enumerate(statements):
#             logger.info(f"Processing statement {i + 1}/{len(statements)}")
            
#             # Build context from previous statements
#             context = " ".join(statements[:i]) if i > 0 else None
            
#             # Generate questions
#             questions = run_rarr_question_generation(
#                 claim=statement,
#                 model=model,
#                 prompt=base_prompt,
#                 temperature=temperature,
#                 context=context,
#                 previous_questions=all_previous_questions
#             )
            
#             results[statement] = questions
#             all_previous_questions.extend(questions)
            
#             logger.info(f"Generated {len(questions)} questions for statement {i + 1}")
        
#         return results

#     except Exception as e:
#         logger.error(f"Error processing atomic statements: {e}")
#         return {statement: [
#             "What evidence supports this statement?",
#             "What sources can verify this information?",
#             "How can these claims be validated?"
#         ] for statement in statements}

# if __name__ == "__main__":
#     # Example usage
#     example_statements = [
#         "Lena Headey portrayed Cersei Lannister in HBO's Game of Thrones since 2011.",
#         "She received three consecutive Emmy nominations for the role.",
#         "She became one of the highest-paid television actors by 2017."
#     ]
    
#     example_prompt = """Generate specific questions to verify the following information."""
    
#     results = process_atomic_statements(
#         statements=example_statements,
#         model="gpt-3.5-turbo",
#         base_prompt=example_prompt,
#         temperature=0.7
#     )
    
#     print("\nGenerated Questions per Statement:")
#     for statement, questions in results.items():
#         print(f"\nStatement: {statement}")
#         for i, question in enumerate(questions, 1):
#             print(f"{i}. {question}")



# # """Utils for running question generation."""
# # import os
# # import time
# # from typing import List
# # import openai

# # openai.api_key = os.getenv("OPENAI_API_KEY")


# # def parse_api_response(api_response: str) -> List[str]:
# #     """Extract questions from the GPT API response.

# #     Args:
# #         api_response: Question generation response from GPT.
# #     Returns:
# #         questions: A list of questions.
# #     """
# #     questions = []
# #     for line in api_response.split("\n"):
# #         line = line.strip()
# #         # Match lines that start with a number (e.g., "1. ", "2)")
# #         if line and line[0].isdigit():
# #             # Extract question text after the number and optional punctuation
# #             split_index = line.find(" ")  # Find the first space
# #             if split_index != -1:
# #                 question = line[split_index + 1:].strip()
# #                 questions.append(question)
# #     return questions


# # def run_rarr_question_generation(
# #     claim: str,
# #     model: str,
# #     prompt: str,
# #     temperature: float = 0.7,
# #     num_rounds: int = 1,
# #     context: str = None,
# #     num_retries: int = 5,
# # ) -> List[str]:
# #     """Generates questions that interrogate the information in a claim.

# #     Args:
# #         claim: Text to generate questions for.
# #         model: Name of the OpenAI GPT model to use.
# #         prompt: The prompt template to query GPT with.
# #         temperature: Temperature for GPT sampling. 0 for deterministic output.
# #         num_rounds: Number of times to sample questions for diversity.
# #         context: Optional context to include with the claim.
# #         num_retries: Number of retries in case of API errors.
# #     Returns:
# #         questions: A list of unique questions.
# #     """
# #     if context:
# #         gpt_input = prompt.format(context=context, claim=claim).strip()
# #     else:
# #         gpt_input = prompt.format(claim=claim).strip()

# #     questions = set()

# #     for _ in range(num_rounds):
# #         for attempt in range(num_retries):
# #             try:
# #                 response = openai.ChatCompletion.create(
# #                     model=model,
# #                     messages=[
# #                         {"role": "system", "content": "You are a helpful assistant."},
# #                         {"role": "user", "content": gpt_input},
# #                     ],
# #                     temperature=temperature,
# #                     max_tokens=256,
# #                 )
# #                 # Parse response for questions
# #                 cur_round_questions = parse_api_response(response.choices[0].message["content"].strip())
# #                 questions.update(cur_round_questions)
# #                 break  # Exit retry loop on success
# #             except openai.error.OpenAIError as exception:
# #                 print(f"API Error: {exception}. Retrying... ({attempt + 1}/{num_retries})")
# #                 time.sleep(1)  # Wait before retrying
# #         else:
# #             print(f"Failed to generate questions after {num_retries} retries.")

# #     return list(sorted(questions))


# # if __name__ == "__main__":
# #     # Example usage (for testing purposes)
# #     example_claim = "Albert Einstein developed the theory of relativity."
# #     example_prompt = """You said: {claim}
# #     To verify, I will ask questions:
# #     1. Who developed the theory of relativity?
# #     2. What is the theory of relativity about?"""

# #     questions = run_rarr_question_generation(
# #         claim=example_claim,
# #         model="gpt-3.5-turbo",
# #         prompt=example_prompt,
# #         temperature=0.7,
# #         num_rounds=3,
# #     )
# #     print("Generated Questions:")
# #     for question in questions:
# #         print(f"- {question}")






# # """Utils for running question generation."""
# # import os
# # import time
# # from typing import List

# # import openai

# # # Set up OpenAI API key from environment variable
# # openai.api_key = os.getenv("OPENAI_API_KEY")

# # def parse_api_response(api_response: str) -> List[str]:
# #     """Extracts questions from the model's response.

# #     The prompt returns questions as a string in an ordered list format.
# #     This function parses the response and extracts individual questions.

# #     Args:
# #         api_response: Question generation response from the model.
# #     Returns:
# #         questions: A list of parsed questions.
# #     """
# #     search_string = "I googled:"
# #     questions = []
# #     for question in api_response.split("\n"):
# #         # Only consider lines that contain the search string
# #         if search_string in question:
# #             question = question.split(search_string)[1].strip()
# #             questions.append(question)
# #     return questions

# # def run_rarr_question_generation(
# #     claim: str,
# #     model: str,
# #     prompt: str,
# #     temperature: float,
# #     num_rounds: int,
# #     context: str = None,
# #     num_retries: int = 5,
# # ) -> List[str]:
# #     """Generates questions that interrogate the information in a claim.

# #     Given a claim, this function uses the model to generate questions that test or
# #     clarify the information within the claim. It samples questions multiple times 
# #     to achieve diversity.

# #     Args:
# #         claim: Text from which to generate questions.
# #         model: Name of the OpenAI model to use (e.g., "gpt-3.5-turbo" or "text-davinci-003").
# #         prompt: The prompt template to query the model with.
# #         temperature: Temperature for sampling. 0 represents greedy decoding.
# #         num_rounds: Number of times to sample questions.
# #         context: Optional context to include with the claim.
# #         num_retries: Number of retries in case of API failures.
# #     Returns:
# #         questions: A list of generated questions.
# #     """
# #     # Format the input prompt with optional context
# #     if context:
# #         formatted_input = prompt.format(context=context, claim=claim).strip()
# #     else:
# #         formatted_input = prompt.format(claim=claim).strip()

# #     questions = set()
# #     for _ in range(num_rounds):
# #         for _ in range(num_retries):
# #             try:
# #                 # Check if the model is a chat model (e.g., gpt-3.5-turbo)
# #                 if "gpt-3.5-turbo" in model or "gpt-4" in model:
# #                     response = openai.ChatCompletion.create(
# #                         model=model,
# #                         messages=[{"role": "user", "content": formatted_input}],
# #                         temperature=temperature,
# #                         max_tokens=256,
# #                     )
# #                     response_text = response.choices[0].message["content"].strip()
# #                 else:
# #                     # Use regular Completion endpoint for non-chat models
# #                     response = openai.Completion.create(
# #                         model=model,
# #                         prompt=formatted_input,
# #                         temperature=temperature,
# #                         max_tokens=256,
# #                     )
# #                     response_text = response.choices[0].text.strip()

# #                 # Parse and add questions from the current response
# #                 cur_round_questions = parse_api_response(response_text)
# #                 questions.update(cur_round_questions)
# #                 break

# #             except openai.error.OpenAIError as exception:
# #                 print(f"{exception}. Retrying...")
# #                 time.sleep(1)

# #     # Return the sorted list of unique questions
# #     return list(sorted(questions))
