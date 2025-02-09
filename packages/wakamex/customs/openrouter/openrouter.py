# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Mihai Cosma
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""OpenRouter request tool."""

import logging
import random
import time
from typing import Any, Dict, Literal, Optional, Tuple
import requests
import json

import openai

# Constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_RETRY_MULTIPLIER = 1.5
DEFAULT_MAX_DELAY = 60.0
DEFAULT_MODEL = "openai/gpt-4o-2024-11-20"
DEFAULT_TEMPERATURE = 1.0

logger = logging.getLogger(__name__)

def with_retries(func):
    """Decorator for retrying functions with exponential backoff."""
    def wrapper(*args, **kwargs):
        max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
        initial_delay = kwargs.pop('initial_delay', DEFAULT_INITIAL_DELAY)
        retry_multiplier = kwargs.pop('retry_multiplier', DEFAULT_RETRY_MULTIPLIER)
        max_delay = kwargs.pop('max_delay', DEFAULT_MAX_DELAY)

        retries = 0
        delay = initial_delay

        while retries <= max_retries:
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.RequestException, 
                   requests.exceptions.HTTPError,
                   requests.exceptions.ConnectionError,
                   requests.exceptions.Timeout,
                   ValueError,
                   json.JSONDecodeError) as exc:
                retries += 1
                if retries > max_retries:
                    logger.error("Max retries (%s) exceeded for %s: %s", max_retries, func.__name__, exc)
                    raise exc
                logger.warning("Attempt %d/%d failed for %s: %s", retries, max_retries, func.__name__, exc)

                # Calculate next delay with exponential backoff and jitter
                multiplier = retry_multiplier
                multiplier *= random.uniform(0.9, 1.1)  # Add jitter to multiplier up to 10%
                delay = min(delay * multiplier, max_delay)
                delay += random.uniform(0, 0.1 * delay)  # Add jitter up to 10%

                logger.warning("Error in %s (attempt %s/%s): %s, retrying in %.2f seconds...", func.__name__, retries, max_retries, str(exc), delay)
                time.sleep(delay)

    return wrapper

@with_retries
def _get_model_response(
        model: str,
        prompt: str,
        api_key: str,
        temperature: float | None = DEFAULT_TEMPERATURE,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        response_format: Literal["text", "json_object"] | None = None
    ) -> Dict:
    """Get a response from a model."""
    logger.debug("Getting response from %s", model)

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://registry.olas.network/ethereum/components/ENTER_MINTED_COMPONENT_NUMBER_HERE",
            "X-Title": "Autonolas",
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        **dict((k, v) for (k, v) in [
            ("frequency_penalty", frequency_penalty),
            ("presence_penalty", presence_penalty),
            ("logprobs", logprobs),
            ("max_tokens", max_tokens),
            ("response_format", response_format)
        ] if v is not None)
    )
    response_text = response.choices[0].message.content
    logger.debug("Raw response length from %s: %s chars", model, len(response_text))

    return response_text

def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task.

    Args:
        **kwargs: Keyword arguments:
            - prompt: The prompt to send to the model
            - model: Optional model to use (defaults to DEFAULT_MODEL)
            - api_keys: Dict containing "openrouter" API key
            - max_retries: Optional max retry attempts
            - initial_delay: Optional initial retry delay
            - retry_multiplier: Optional retry delay multiplier
            - max_delay: Optional maximum retry delay

    Returns:
        Tuple containing:
        - Response to send to the user
        - Optional prompt sent to the model
        - Optional transaction generated by the tool
        - Optional cost calculation object
    """
    # Get required parameters
    prompt = kwargs.get("prompt")
    if not prompt:
        return "No prompt has been specified.", None, None, None

    api_keys = kwargs.get("api_keys", {})
    api_key = api_keys.get("openrouter")
    if not api_key:
        return "No OpenRouter API key has been specified.", None, None, None

    # Get optional parameters
    model = kwargs.get("model", DEFAULT_MODEL)

    try:
        response = _get_model_response(
            model=model,
            prompt=prompt,
            api_key=api_key,
            max_retries=kwargs.get("max_retries", DEFAULT_MAX_RETRIES),
            initial_delay=kwargs.get("initial_delay", DEFAULT_INITIAL_DELAY),
            retry_multiplier=kwargs.get("retry_multiplier", DEFAULT_RETRY_MULTIPLIER),
            max_delay=kwargs.get("max_delay", DEFAULT_MAX_DELAY)
        )

        return response, None, None, None

    except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
        error_msg = f"Error while calling OpenRouter API: {str(exc)}"
        logger.error(error_msg)
        return error_msg, None, None, None
