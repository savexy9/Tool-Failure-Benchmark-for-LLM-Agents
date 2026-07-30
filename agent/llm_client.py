"""
Model-agnostic LLM client using the OpenAI-compatible chat completions API.

Default provider: NVIDIA NIM (https://integrate.api.nvidia.com/v1).
Swap to any OpenAI-compatible provider by changing base_url.
"""

from __future__ import annotations

import time

from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError

from config.settings import LLMConfig


class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_retries: int = 8,
    ) -> str:
        """Send a chat completion request with retry on rate-limit and timeout errors.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Override the default temperature if set.
            max_retries: Maximum number of retries on transient errors.

        Returns:
            The assistant's reply as a string.
        """
        temp = temperature if temperature is not None else self.config.temperature

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.config.max_tokens,
                )
                msg = response.choices[0].message
                # Handle reasoning models (e.g., Nemotron) that put text in reasoning_content
                content = msg.content or msg.reasoning_content or ""
                return content
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80, 160, 320s
                err_type = "rate-limited" if isinstance(e, RateLimitError) else "timeout"
                print(f"    [{err_type}] retry {attempt + 1}/{max_retries} in {wait}s...")
                time.sleep(wait)
            except APIStatusError as e:
                # Retry on 429 (rate limit), 500 (internal), 502 (bad gateway), 503 (overloaded), 529 (overloaded)
                if e.status_code in (429, 500, 502, 503, 529) and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"    [HTTP {e.status_code}] retry {attempt + 1}/{max_retries} in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
