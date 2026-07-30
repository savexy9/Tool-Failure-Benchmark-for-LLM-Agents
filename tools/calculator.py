"""
Calculator tool with fault injection.

Evaluates arithmetic expressions safely using eval() with restricted builtins.
Supports four fault modes: silent_wrong, error, malformed, empty.
"""

from __future__ import annotations

import random as _random


def tool_calculator(expression: str, fault: str | None = None) -> str:
    """Evaluate an arithmetic expression, optionally injecting a fault.

    Args:
        expression: A Python arithmetic expression (e.g., "2 + 3 * 4").
        fault: Fault type to inject. One of None, "silent_wrong", "error",
               "malformed", "empty".

    Returns:
        String representation of the result, or a fault-injected output.

    Raises:
        RuntimeError: When fault == "error".
    """
    if fault == "error":
        raise RuntimeError("Calculator service timeout")

    if fault == "malformed":
        return "###ERR_9f2@@"

    if fault == "empty":
        return ""

    try:
        result = eval(expression, {"__builtins__": {}})
    except Exception:
        return "Error: invalid expression"

    if fault == "silent_wrong":
        # Return a plausible but wrong answer: offset by a small random amount
        try:
            numeric = float(result)
            offset = _random.choice([1, -1, 10, -10, 0.5, -0.5])
            return str(numeric + offset)
        except (TypeError, ValueError):
            return "42"  # fallback wrong answer

    return str(result)
