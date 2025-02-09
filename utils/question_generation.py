"""
question_generation.py

Module for generating fact-checking questions for atomic statements in the RARR framework.
"""

import time
import logging
from typing import List, Dict, Optional
from openai.error import OpenAIError
from utils.newfile import call_openai_api

from mistralai import Mistral
# Initialize Mistral API
api_key = "Replace with your valid API key"  # Replace with your valid API key
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

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
                #client.chat.complete
                try:
                    response = client.chat.complete(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": gpt_input}
                        ],
                        model=model,
                        temperature=temperature,
                    )
                    
                    raw_text = response.choices[0].message.content
                    #print("QUESTION GENERATION.py file's question: ", raw_text)
                    
                    new_questions = parse_api_response(raw_text, required_questions)

                    for q in new_questions:
                        #print("this is the question that was generated: ", q)
                        all_questions.add(q)

                    break

                except Mistral.APIError as e:
                    logger.warning(f"OpenAI API Error (attempt {attempt_i + 1}/{num_retries}): {e}")
                    if attempt_i < num_retries - 1:
                        time.sleep(2)
                except Exception as e:
                    logger.warning(f"Unexpected Error (attempt {attempt_i + 1}/{num_retries}):")
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
            if not questions:
                #print("It's EMPTYYYY")


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
        model="mistral-large-latest",
        base_prompt=base_prompt,
        temperature=0.7
    )

    for statement, questions in results.items():
        print(f"\nStatement: {statement}")
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
