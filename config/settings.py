"""
Configuration management for the Tool-Failure Reliability experiment.

All experiment parameters are centralized here as dataclasses.
CLI arguments override defaults; environment variables override CLI for secrets.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from pathlib import Path


# ── Fault types ──────────────────────────────────────────────────────────────

FAULT_TYPES: list[str | None] = [None, "silent_wrong", "error", "malformed", "empty"]
FAULT_LABELS: dict[str | None, str] = {
    None: "none",
    "silent_wrong": "silent_wrong",
    "error": "error",
    "malformed": "malformed",
    "empty": "empty",
}

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
DATASET_PATH = PROJECT_ROOT / "evaluation_dataset.csv"


@dataclass(frozen=True)
class LLMConfig:
    """Settings for the LLM API client."""

    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: str = field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY", ""))
    model: str = "meta/llama-3.1-70b-instruct"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 120.0


@dataclass(frozen=True)
class FaultConfig:
    """Settings for fault injection behaviour."""

    fault_types: tuple[str | None, ...] = field(
        default_factory=lambda: tuple(FAULT_TYPES)
    )
    # Deterministic seed for fault scheduling (separate from LLM randomness)
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    prompt_conditions: tuple[bool, bool] = (False, True)  # (no-verify, verify)
    dataset_path: Path = DATASET_PATH
    output_csv: Path = DATA_DIR / "results.csv"
    figures_dir: Path = FIGURES_DIR
    # Global random seed (controls Python stdlib, numpy if available)
    seed: int = 42
    # If set, skip experiment and only regenerate charts from this CSV
    analyze_only: str | None = None


# ── CLI Parsing ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    """Parse CLI arguments and return a frozen ExperimentConfig."""

    parser = argparse.ArgumentParser(
        description="LLM Agent Tool-Failure Reliability Benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta/llama-3.1-70b-instruct",
        help="Model identifier for the LLM provider (default: meta/llama-3.1-70b-instruct)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://integrate.api.nvidia.com/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help="Path to evaluation dataset CSV (default: evaluation_dataset.csv)",
    )
    parser.add_argument(
        "--analyze-only",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help="Skip experiment; regenerate charts from existing CSV",
    )

    args = parser.parse_args(argv)

    # Apply global random seed
    random.seed(args.seed)

    llm = LLMConfig(
        base_url=args.base_url,
        model=args.model,
    )
    fault = FaultConfig(seed=args.seed)

    dataset_path = Path(args.dataset) if args.dataset else DATASET_PATH

    # Use model name in output path to avoid overwriting
    model_slug = args.model.replace("/", "_")
    output_csv = DATA_DIR / f"results_{model_slug}.csv"

    config = ExperimentConfig(
        llm=llm,
        fault=fault,
        seed=args.seed,
        analyze_only=args.analyze_only,
        dataset_path=dataset_path,
        output_csv=output_csv,
    )

    # Ensure output directories exist
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    return config
