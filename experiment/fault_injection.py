"""
Fault injection framework.

Provides deterministic fault scheduling for reproducible experiments.
Each test case is paired with each fault type exactly once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from config.settings import FAULT_TYPES


@dataclass
class TrialSpec:
    """Specification for a single trial to be executed."""

    trial_id: int  # Unique ID in the full trial matrix
    test_case_id: int
    tool: str
    question: str
    tool_input: str
    ground_truth: str
    difficulty: str
    fault_type: str | None
    prompt_condition: bool  # False = no verify, True = verify prompt


def build_trial_matrix(
    test_cases: list,
    fault_types: list[str | None] | None = None,
    prompt_conditions: tuple[bool, bool] = (False, True),
    seed: int = 42,
) -> list[TrialSpec]:
    """Build the full matrix of trials: test_cases × faults × prompt conditions.

    Args:
        test_cases: List of TestCase objects.
        fault_types: List of fault types to inject. Defaults to FAULT_TYPES.
        prompt_conditions: Tuple of (no-verify, verify) booleans.
        seed: Random seed for any randomized ordering (not used for fault assignment).

    Returns:
        List of TrialSpec objects in deterministic order.
    """
    if fault_types is None:
        fault_types = FAULT_TYPES

    rng = random.Random(seed)
    trials: list[TrialSpec] = []
    trial_id = 0

    for tc in test_cases:
        for fault in fault_types:
            for verify in prompt_conditions:
                trial_id += 1
                trials.append(
                    TrialSpec(
                        trial_id=trial_id,
                        test_case_id=tc.id,
                        tool=tc.tool,
                        question=tc.question,
                        tool_input=tc.tool_input,
                        ground_truth=tc.ground_truth,
                        difficulty=tc.difficulty,
                        fault_type=fault,
                        prompt_condition=verify,
                    )
                )

    # Shuffle deterministically for realistic timing (API latency variance)
    rng.shuffle(trials)
    return trials
