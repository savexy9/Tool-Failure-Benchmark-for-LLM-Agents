# Evaluation Dataset — LLM Agent Tool-Failure Reliability Benchmark

## Overview

This dataset contains **66 test cases** across three tool categories, designed to evaluate LLM agent reliability when tools return incorrect, malformed, or missing results.

- **20** calculator questions
- **24** Wikipedia factual lookup questions
- **22** unit conversion questions

Each row includes a natural-language question, the tool input to be passed to the corresponding tool function, a manually verified ground-truth answer, and a difficulty rating.

## Dataset Construction

### Design Principles

1. **Unambiguous answers**: Every question has exactly one correct ground-truth answer. No opinion-based, time-sensitive, or disputed facts.
2. **Stable over time**: Questions reference historical facts, mathematical constants, and established scientific knowledge. No current events, rankings, or statistics that change frequently.
3. **Verifiable**: All ground-truth values were manually cross-checked against primary sources.
4. **Difficulty balance**: Each tool category has questions rated easy, medium, and hard to test robustness across complexity levels.
5. **tool_input clarity**: The `tool_input` column contains the exact string to pass to the tool function, formatted to maximize successful tool execution.

### Difficulty Definitions

| Level | Calculator | Lookup | Converter |
|-------|-----------|--------|-----------|
| **Easy** | Single-operation arithmetic (add, subtract, multiply, divide) | Well-known facts about globally recognized entities | Common unit pairs (°F↔°C, mi↔km, lb↔kg) |
| **Medium** | Multi-step expressions, percentages, powers, order of operations | Moderately specific historical/scientific facts | Decimal values, less common unit pairs |
| **Hard** | Nested parentheses, exponents, compound expressions | Technical or specialized knowledge requiring precise retrieval | Less common units (stone, nautical miles, Rankine, hectares) |

### Per-Tool Breakdown

#### Calculator (20 questions)

- **Easy (7)**: Basic arithmetic — addition, subtraction, multiplication, division, simple grouping.
- **Medium (7)**: Percentages, powers, multi-step order of operations.
- **Hard (6)**: Nested expressions, exponents, fractional multipliers.

`tool_input` is a Python arithmetic expression safe for `eval()` with restricted builtins.

#### Wikipedia Lookup (24 questions)

- **Easy (7)**: Globally known facts (planets, capitals, famous people/artworks).
- **Medium (7)**: Specific dates, scientific quantities, named discoveries.
- **Hard (10)**: Technical concepts, mathematical constants, thought experiments.

`tool_input` is the exact Wikipedia page title (with spaces, matching the REST API `/page/summary/{title}` endpoint). Answers were verified against Wikipedia article summaries as of dataset creation.

#### Unit Conversion (22 questions)

- **Easy (7)**: Standard conversions with clean numbers (°F↔°C, mi↔km, lb↔kg, cm↔in, gal↔L, oz↔g).
- **Medium (7)**: Decimal values, meter-to-mile, gram-to-pound conversions.
- **Hard (8)**: Kelvin to Fahrenheit/Rankine, hectares to acres, stone to pounds, nautical miles to kilometers.

`tool_input` is formatted as `"{value} {unit_abbreviation}"`. Ground-truth values are computed using standard conversion factors and rounded to 4 decimal places.

## Ground-Truth Verification

### Calculator

All expressions were evaluated by hand and cross-checked with Python:
```python
eval(expression, {"__builtins__": {}})
```

### Lookup

Answers verified against Wikipedia article summaries. Key facts:
- Speed of light: 299,792,458 m/s (exact, by definition since 1983)
- Standard gravity: 9.80665 m/s² (exact, by CGPM definition)
- French Revolution: began 1789 (Storming of the Bastille, July 14)
- Schrödinger's cat: proposed 1935, illustrates quantum superposition

### Converter

Conversion factors used:
| From | To | Factor |
|------|-----|--------|
| °F | °C | (F - 32) × 5/9 |
| °C | °F | C × 9/5 + 32 |
| mi | km | 1.609344 |
| lb | kg | 0.453592 |
| cm | in | 0.393701 |
| gal (US) | L | 3.78541 |
| oz | g | 28.3495 |
| m | mi | 0.000621371 |
| g | lb | 0.00220462 |
| K | °F | (K - 273.15) × 9/5 + 32 |
| ha | acre | 2.47105 |
| K | °R | K × 1.8 |
| st | lb | 14 |
| nmi | km | 1.852 |

## CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique identifier (1-66) |
| `tool` | str | Tool category: `calculator`, `lookup`, or `converter` |
| `question` | str | Natural-language question posed to the agent |
| `tool_input` | str | Exact input to pass to the tool function |
| `ground_truth` | str | Expected correct answer (for automated evaluation) |
| `difficulty` | str | `easy`, `medium`, or `hard` |

## Usage

Load the dataset in Python:
```python
import csv

with open("evaluation_dataset.csv", newline="") as f:
    reader = csv.DictReader(f)
    test_cases = list(reader)
```

Filter by tool or difficulty:
```python
calculator_easy = [r for r in test_cases if r["tool"] == "calculator" and r["difficulty"] == "easy"]
```

## Assumptions

1. **Wikipedia availability**: Lookup questions assume the Wikipedia REST API is accessible and page titles match the `tool_input` values. Some titles may redirect or require exact casing.
2. **Converter scope**: The converter tool handles single-step unit conversions. Multi-step conversions (e.g., °F → K) are treated as single operations with the appropriate formula.
3. **Ground-truth matching**: Evaluation should use flexible matching (contains, numeric tolerance) rather than exact string comparison, especially for lookup answers which may vary in phrasing.
4. **Static dataset**: This dataset is a snapshot. Wikipedia content may change over time, potentially affecting lookup ground truths for non-historical facts.
