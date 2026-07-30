"""Tool implementations with configurable fault injection."""

from .calculator import tool_calculator
from .lookup import tool_lookup
from .converter import tool_converter

TOOL_REGISTRY: dict[str, callable] = {
    "calculator": tool_calculator,
    "lookup": tool_lookup,
    "converter": tool_converter,
}

__all__ = ["tool_calculator", "tool_lookup", "tool_converter", "TOOL_REGISTRY"]
