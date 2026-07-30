"""
Wikipedia lookup tool with fault injection.

Fetches article summaries via the Wikipedia REST API.
Supports four fault modes: silent_wrong, error, malformed, empty.
"""

from __future__ import annotations

import requests


WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"


def tool_lookup(query: str, fault: str | None = None) -> str:
    """Look up a Wikipedia article summary, optionally injecting a fault.

    Args:
        query: Wikipedia page title (spaces allowed, will be converted to underscores).
        fault: Fault type to inject.

    Returns:
        Article extract text, or a fault-injected output.

    Raises:
        RuntimeError: When fault == "error".
    """
    if fault == "error":
        raise RuntimeError("Lookup service unavailable")

    if fault == "malformed":
        return "<<garbled_response_0x00>>"

    if fault == "empty":
        return ""

    if fault == "silent_wrong":
        return f"{query} was founded in 1850 in a small unrelated town."

    try:
        url = WIKIPEDIA_API + query.replace(" ", "_")
        resp = requests.get(url, timeout=10, headers={"User-Agent": "ToolReliabilityBenchmark/1.0"})
        resp.raise_for_status()
        data = resp.json()
        return data.get("extract", "No summary found.")
    except requests.RequestException:
        return "Error: lookup failed"
