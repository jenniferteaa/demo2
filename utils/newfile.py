import os
import time
import json
import hashlib
import openai

from openai.error import RateLimitError, ServiceUnavailableError

# If OPENAI_API_KEY isn't already set in openai.api_key, try loading from environment


from mistralai import Mistral
# Initialize Mistral API
api_key = "Replace with your valid API key"  # 
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

def call_openai_api(
    messages,
    model="mistral-large-latest",
    temperature=0,
    max_retries=5,
    use_cache=True,
    cache_dir="cache"
):
    """
    A wrapper for openai.ChatCompletion.create that implements:
      - Local file-based caching
      - Retry/backoff logic on rate-limit

    Args:
        messages (List[Dict]): e.g. [{"role": "user", "content": "..."}]
        model (str): OpenAI model name (default "gpt-3.5-turbo").
        temperature (float): Sampling temperature (default 0).
        max_retries (int): Times to retry on rate limits or service errors.
        use_cache (bool): Whether to cache the request/responses locally.
        cache_dir (str): Directory to store cached JSON responses.

    Returns:
        Dict: The full response from OpenAI ChatCompletion.

    Raises:
        Exception: If max_retries are exhausted or another error occurs.
    """
    # 1. Check local cache first
    if use_cache:
        cached_response = get_cached_response(messages, model, temperature, cache_dir)
        if cached_response is not None:
            return cached_response

    # 2. Attempt the API call with retries/backoff
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature
            )
            print("response from newfile: ", response)
            # 3. Store response in cache, if successful and cache is enabled
            if use_cache:
                save_to_cache(messages, model, temperature, response, cache_dir)
            return response

        except RateLimitError as e:
            print(f"[call_openai_api] RateLimitError: {e}. Attempt {attempt+1}/{max_retries}")
            backoff = 10 * (attempt + 1)
            print(f" -> Sleeping for {backoff} seconds...")
            time.sleep(backoff)
            attempt += 1

        except ServiceUnavailableError as e:
            print(f"[call_openai_api] ServiceUnavailableError: {e}. Attempt {attempt+1}/{max_retries}")
            backoff = 5 * (attempt + 1)
            print(f" -> Sleeping for {backoff} seconds...")
            time.sleep(backoff)
            attempt += 1

    # If we got here, we never succeeded
    raise Exception("Max retries reached. OpenAI API call failed.")

def get_cached_response(messages, model, temperature, cache_dir):
    """
    Check if there's a cached response for the given (messages, model, temperature).
    Returns the response (dict) if it exists, else None.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_key = create_cache_key(messages, model, temperature)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        return cached_data
    return None

def save_to_cache(messages, model, temperature, response, cache_dir):
    """
    Save the response to a local JSON file, keyed by the hash of (messages+model+temperature).
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_key = create_cache_key(messages, model, temperature)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    # Convert 'OpenAIObject' to dict if needed
    if hasattr(response, 'to_dict_recursive'):
        response = response.to_dict_recursive()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(response, f)

def create_cache_key(messages, model, temperature):
    """
    Create a hash key based on the prompt. The order & content of messages matter.
    """
    raw_data = {
        "model": model,
        "temperature": temperature,
        "messages": messages
    }
    encoded = json.dumps(raw_data, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()
