"""
Tool-Failure Reliability Benchmark — Main Entry Point

Runs the full experiment:
  1. Load test cases from evaluation_dataset.csv
  2. Build trial matrix (queries × faults × prompt conditions)
  3. Execute trials with the configured LLM
  4. Log results to CSV
  5. Compute metrics and generate charts

Usage:
    python main.py                                          # default: meta/llama-3.1-70b-instruct, seed=42
    python main.py --model meta/llama-3.1-8b-instruct      # different model
    python main.py --seed 123                               # different seed
    python main.py --analyze-only data/results.csv          # regenerate charts from existing CSV

Environment:
    NVIDIA_API_KEY  — required for NVIDIA NIM (or any OpenAI-compatible provider)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from config.settings import parse_args
from experiment.test_cases import load_test_cases
from experiment.fault_injection import build_trial_matrix
from experiment.runner import ExperimentRunner
from evaluation.metrics import compute_all_metrics
from visualization.charts import generate_all_charts


def run_experiment(config) -> None:
    """Run the full experiment pipeline."""
    # 1. Load test cases
    print(f"Loading test cases from {config.dataset_path}")
    test_cases = load_test_cases(config.dataset_path)
    print(f"  Loaded {len(test_cases)} test cases")

    # 2. Build trial matrix
    trials = build_trial_matrix(
        test_cases,
        fault_types=list(config.fault.fault_types),
        prompt_conditions=config.prompt_conditions,
        seed=config.fault.seed,
    )
    print(f"  Built trial matrix: {len(trials)} trials")

    # 3. Check for API key
    if not config.llm.api_key:
        print("\nERROR: NVIDIA_API_KEY environment variable not set.")
        print("  export NVIDIA_API_KEY='nvapi-...'")
        sys.exit(1)

    # 4. Run trials
    runner = ExperimentRunner(config)
    results = runner.run_all(trials)

    # 5. Compute and display metrics
    metrics = compute_all_metrics(results)
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"  Total trials:        {metrics.total_trials}")
    print(f"  Clean trials:        {metrics.clean_trials}")
    print(f"  Faulted trials:      {metrics.faulted_trials}")
    print(f"  Accuracy (overall):  {metrics.accuracy_overall:.1%}")
    print(f"  Accuracy (clean):    {metrics.accuracy_clean:.1%}")
    print(f"  Accuracy (faulted):  {metrics.accuracy_faulted:.1%}")
    print(f"  Detection rate:      {metrics.detection_rate:.1%}")
    print(f"  Silent failure rate: {metrics.silent_failure_rate:.1%}")
    print(f"  Recovery rate:       {metrics.recovery_rate:.1%}")
    print("=" * 60)

    # 6. Generate charts
    generate_all_charts(results, config.figures_dir)


def analyze_only(csv_path: str, figures_dir: Path) -> None:
    """Regenerate charts from an existing results CSV."""
    print(f"Loading results from {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = list(reader)

    # Convert types
    for r in results:
        r["is_correct"] = r["is_correct"] in ("True", "true", "1")
        r["agent_flagged"] = r["agent_flagged"] in ("True", "true", "1")
        r["latency_ms"] = float(r.get("latency_ms", 0))

    metrics = compute_all_metrics(results)
    print(f"\n  Loaded {len(results)} trials")
    print(f"  Accuracy (overall):  {metrics.accuracy_overall:.1%}")
    print(f"  Detection rate:      {metrics.detection_rate:.1%}")
    print(f"  Silent failure rate: {metrics.silent_failure_rate:.1%}")

    generate_all_charts(results, figures_dir)


def main() -> None:
    """CLI entry point."""
    config = parse_args()

    if config.analyze_only:
        analyze_only(config.analyze_only, config.figures_dir)
    else:
        run_experiment(config)


if __name__ == "__main__":
    main()
