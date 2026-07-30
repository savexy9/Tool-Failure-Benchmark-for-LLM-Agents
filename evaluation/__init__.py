"""Evaluation module: correctness checking and metric computation."""

from .metrics import (
    evaluate_correctness,
    detect_fault_flagged,
    compute_metrics,
    compute_all_metrics,
)

__all__ = [
    "evaluate_correctness",
    "detect_fault_flagged",
    "compute_metrics",
    "compute_all_metrics",
]
