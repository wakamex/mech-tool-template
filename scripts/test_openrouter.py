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
"""Script to test the OpenRouter tool."""

from dotenv import load_dotenv
from packages.wakamex.customs.openrouter.openrouter import run
import os

if __name__ == "__main__":
    # Load the API key
    load_dotenv()
    openrouter_api_key = os.environ["OPENROUTER_API_KEY"]

    # Call the tool
    result = run(
        prompt="Write a haiku about coding",
        api_keys={"openrouter": openrouter_api_key}
    )

    # Print the result
    print(f"Result: {result}")
