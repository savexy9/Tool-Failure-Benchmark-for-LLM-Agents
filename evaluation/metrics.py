"""
Evaluation metrics for the Tool-Failure Reliability experiment.

Provides:
  - Per-trial correctness evaluation (rule-based)
  - Per-trial fault detection
  - Aggregate metric computation: silent failure rate, detection rate,
    recovery rate, accuracy
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ── Per-trial evaluation ─────────────────────────────────────────────────────


def evaluate_correctness(
    agent_answer: str,
    ground_truth: str,
    tool: str,
) -> bool:
    """Check whether the agent's answer matches the ground truth.

    Uses tool-specific matching strategies:
      - calculator: numeric tolerance (parse both as float, ±0.01)
      - converter: numeric tolerance (extract number from answer, ±1%)
      - lookup: keyword containment (ground truth tokens found in answer)

    Args:
        agent_answer: The agent's final answer text.
        ground_truth: The expected correct answer.
        tool: The tool category ("calculator", "lookup", "converter").

    Returns:
        True if the answer is considered correct.
    """
    if not agent_answer or not ground_truth:
        return False

    answer_lower = agent_answer.lower().strip()
    truth_lower = ground_truth.lower().strip()

    if tool == "calculator":
        return _numeric_match(answer_lower, truth_lower, tolerance=0.01)

    if tool == "converter":
        return _numeric_match(answer_lower, truth_lower, tolerance_pct=0.01)

    if tool == "lookup":
        return _keyword_match(answer_lower, truth_lower)

    # Fallback: substring containment
    return truth_lower in answer_lower


def _numeric_match(answer: str, truth: str, tolerance: float = 0.0, tolerance_pct: float = 0.0) -> bool:
    """Check if a number in the answer matches the ground truth within tolerance."""
    # Try to extract a number from the answer
    answer_nums = _extract_numbers(answer)
    truth_nums = _extract_numbers(truth)

    if not truth_nums:
        return truth.lower() in answer.lower()

    target = truth_nums[0]

    for num in answer_nums:
        if tolerance_pct > 0:
            if abs(num - target) <= abs(target) * tolerance_pct:
                return True
        elif tolerance > 0:
            if abs(num - target) <= tolerance:
                return True
        else:
            if num == target:
                return True

    return False


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a text string."""
    numbers = []
    # Match integers and floats, including negative
    for m in re.finditer(r"-?\d+\.?\d*(?:e[+-]?\d+)?", text):
        try:
            numbers.append(float(m.group()))
        except ValueError:
            continue
    return numbers


def _keyword_match(answer: str, truth: str) -> bool:
    """Check if key tokens from the ground truth appear in the answer."""
    # Split truth into significant tokens (ignore very short words)
    truth_tokens = [t for t in re.split(r"\W+", truth) if len(t) >= 3]

    if not truth_tokens:
        return truth in answer

    # Require at least 60% of truth tokens to be present
    matches = sum(1 for t in truth_tokens if t in answer)
    return matches / len(truth_tokens) >= 0.6


# ── Fault detection ──────────────────────────────────────────────────────────

# Phrases the agent might use to flag a suspicious tool result
_FLAG_PATTERNS = [
    r"seems?\s+(wrong|off|incorrect|unusual|odd|suspicious)",
    r"error",
    r"malformed",
    r"unreliable",
    r"can'?t\s+verify",
    r"unable\s+to\s+confirm",
    r"not\s+confident",
    r"doesn'?t\s+seem\s+right",
    r"looks?\s+(wrong|incorrect|off|strange|suspicious)",
    r"questionable",
    r"doubt",
    r"unexpected",
    r"garbled",
    r"corrupt",
    r"invalid",
    r"faulty",
    r"incorrect",
    r"inaccurate",
]

_FLAG_RE = re.compile("|".join(_FLAG_PATTERNS), re.IGNORECASE)


def detect_fault_flagged(
    agent_response: str,
    tool_result: str,
    fault_type: str | None,
) -> bool:
    """Determine whether the agent flagged the tool result as suspicious.

    Args:
        agent_response: The agent's full response text.
        tool_result: The tool result that was injected.
        fault_type: The type of fault that was injected (None for clean).

    Returns:
        True if the agent appears to have flagged the issue.
    """
    if fault_type is None or fault_type == "none":
        # For clean trials, we don't expect flagging — return False
        return False

    # Check if the agent's response contains flag-like language
    if _FLAG_RE.search(agent_response):
        # Additional check: make sure the agent isn't just using the word
        # "error" in a different context (e.g., "the calculator returned
        # an error" when the fault was actually "error" type and the agent
        # just accepted it)
        return True

    return False


# ── Aggregate metrics ────────────────────────────────────────────────────────


@dataclass
class Metrics:
    """Computed experiment metrics."""

    total_trials: int = 0
    clean_trials: int = 0
    faulted_trials: int = 0

    # Accuracy
    accuracy_overall: float = 0.0
    accuracy_clean: float = 0.0
    accuracy_faulted: float = 0.0

    # Detection
    detection_rate: float = 0.0  # % of faulted trials where agent flagged

    # Silent failure
    silent_failure_rate: float = 0.0  # % of silent_wrong where agent didn't flag AND was wrong

    # Recovery
    recovery_rate: float = 0.0  # % of flagged trials where answer was still correct

    # Per-fault-type breakdown
    by_fault: dict[str, dict] = None

    def __post_init__(self):
        if self.by_fault is None:
            self.by_fault = {}


def compute_metrics(results: list[dict], fault_filter: str | None = None) -> dict:
    """Compute metrics for a subset of results.

    Args:
        results: List of result dicts from the experiment runner.
        fault_filter: If set, only include trials with this fault type.

    Returns:
        Dict with metric names and values.
    """
    if fault_filter is not None:
        filtered = [r for r in results if r["fault_type"] == fault_filter]
    else:
        filtered = results

    if not filtered:
        return {"count": 0}

    total = len(filtered)
    correct = sum(1 for r in filtered if r["is_correct"])
    flagged = sum(1 for r in filtered if r["agent_flagged"])

    return {
        "count": total,
        "accuracy": correct / total if total else 0.0,
        "detection_rate": flagged / total if total else 0.0,
    }


def compute_all_metrics(results: list[dict]) -> Metrics:
    """Compute all experiment metrics from the results CSV data.

    Args:
        results: List of result dicts (as produced by ExperimentRunner).

    Returns:
        A Metrics object with all computed values.
    """
    m = Metrics()

    if not results:
        return m

    m.total_trials = len(results)

    # Split by fault type
    clean = [r for r in results if r["fault_type"] == "none"]
    faulted = [r for r in results if r["fault_type"] != "none"]

    m.clean_trials = len(clean)
    m.faulted_trials = len(faulted)

    # Accuracy
    m.accuracy_overall = sum(1 for r in results if r["is_correct"]) / m.total_trials
    m.accuracy_clean = (
        sum(1 for r in clean if r["is_correct"]) / len(clean) if clean else 0.0
    )
    m.accuracy_faulted = (
        sum(1 for r in faulted if r["is_correct"]) / len(faulted) if faulted else 0.0
    )

    # Detection rate (across all faulted trials)
    m.detection_rate = (
        sum(1 for r in faulted if r["agent_flagged"]) / len(faulted) if faulted else 0.0
    )

    # Silent failure rate: silent_wrong trials where NOT flagged AND wrong
    silent_wrong = [r for r in results if r["fault_type"] == "silent_wrong"]
    if silent_wrong:
        silent_failures = sum(
            1 for r in silent_wrong
            if not r["agent_flagged"] and not r["is_correct"]
        )
        m.silent_failure_rate = silent_failures / len(silent_wrong)
    else:
        m.silent_failure_rate = 0.0

    # Recovery rate: flagged trials where answer was still correct
    flagged_trials = [r for r in faulted if r["agent_flagged"]]
    if flagged_trials:
        recovered = sum(1 for r in flagged_trials if r["is_correct"])
        m.recovery_rate = recovered / len(flagged_trials)
    else:
        m.recovery_rate = 0.0

    # Per-fault-type breakdown
    fault_types = sorted(set(r["fault_type"] for r in results))
    for ft in fault_types:
        subset = [r for r in results if r["fault_type"] == ft]
        ft_correct = sum(1 for r in subset if r["is_correct"])
        ft_flagged = sum(1 for r in subset if r["agent_flagged"])
        m.by_fault[ft] = {
            "count": len(subset),
            "accuracy": ft_correct / len(subset) if subset else 0.0,
            "detection_rate": ft_flagged / len(subset) if subset else 0.0,
        }

    return m
