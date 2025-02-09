# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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
"""Tests for the OpenRouter tool."""

import logging
from packages.wakamex.customs.openrouter.openrouter import _get_model_response

TEST_MODEL = "meta-llama/llama-3.3-70b-instruct"
TEST_PROMPT = "What is 2+2? Answer with only one character."

def load_dict_from_dotenv() -> dict:
    """Manually parse .env file without libraries."""
    result = {}
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            key, value = line.split("=")
            result[key.strip()] = value.strip()
    return result

dotenv_values = load_dict_from_dotenv()
OPENROUTER_API_KEY = dotenv_values["OPENROUTER_API_KEY"]

def test_basic_response():
    """Test basic model response without provider preferences."""
    response = _get_model_response(TEST_MODEL,TEST_PROMPT,OPENROUTER_API_KEY)
    logging.info(response)
    assert response is not None, "Response is None"
    assert response == '4', "Response is not 4"

def test_provider_sort_by_throughput():
    """Test sorting providers by throughput (e.g., to prefer Groq with 243.8t/s)."""
    response = _get_model_response(
        TEST_MODEL,
        TEST_PROMPT,
        OPENROUTER_API_KEY,
        sort="throughput"
    )
    logging.info(response)
    assert response is not None, "Response is None"
    assert response == '4', "Response is not 4"

def test_provider_sort_by_price():
    """Test sorting by price (e.g., to prefer Lambda at $0.12/input)."""
    response = _get_model_response(
        TEST_MODEL,
        TEST_PROMPT,
        OPENROUTER_API_KEY,
        sort="price"
    )
    logging.info(response)
    assert response is not None, "Response is None"
    assert response == '4', "Response is not 4"

def test_provider_specific_quantization():
    """Test requesting specific quantization (e.g., fp8 providers like Lambda)."""
    response = _get_model_response(
        TEST_MODEL,
        TEST_PROMPT,
        OPENROUTER_API_KEY,
        quantizations=["fp8"]
    )
    logging.info(response)
    assert response is not None, "Response is None"
    assert response == '4', "Response is not 4"

def test_provider_order_with_fallbacks():
    """Test specifying provider order with fallbacks."""
    response = _get_model_response(
        TEST_MODEL,
        TEST_PROMPT,
        OPENROUTER_API_KEY,
        provider_order=["Groq", "Lambda", "DeepInfra"],
        allow_fallbacks=True
    )
    logging.info(response)
    assert response is not None, "Response is None"
    assert response == '4', "Response is not 4"

def test_provider_order_no_fallbacks():
    """Test specifying provider order without fallbacks."""
    response = _get_model_response(
        TEST_MODEL,
        TEST_PROMPT,
        OPENROUTER_API_KEY,
        provider_order=["Lambda", "DeepInfra"],
        allow_fallbacks=False
    )
    logging.info(response)
    assert response is not None, "Response is None"
    assert response == '4', "Response is not 4"

def test_provider_ignore_list():
    """Test ignoring specific providers."""
    response = _get_model_response(
        TEST_MODEL,
        TEST_PROMPT,
        OPENROUTER_API_KEY,
        ignore_providers=["SambaNova", "Cloudflare"]
    )
    logging.info(response)
    assert response is not None, "Response is None"
    assert response == '4', "Response is not 4"

def test_provider_data_collection_deny():
    """Test denying providers that collect data."""
    response = _get_model_response(
        TEST_MODEL,
        TEST_PROMPT,
        OPENROUTER_API_KEY,
        data_collection="deny"
    )
    logging.info(response)
    assert response is not None, "Response is None"
    assert response == '4', "Response is not 4"

def test_provider_require_parameters():
    """Test requiring providers to support all parameters."""
    response = _get_model_response(
        TEST_MODEL,
        TEST_PROMPT,
        OPENROUTER_API_KEY,
        require_parameters=True,
        max_tokens=1000
    )
    logging.info(response)
    assert response is not None, "Response is None"
    assert response == '4', "Response is not 4"
