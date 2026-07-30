"""
Experiment runner: executes trials, logs results to CSV.

Handles the full trial lifecycle:
  1. Inject fault into tool execution
  2. Run the ReAct agent with the (possibly faulted) tool result
  3. Evaluate correctness against ground truth
  4. Log all metadata to CSV (incrementally, so progress is never lost)
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.llm_client import LLMClient
from agent.react_agent import ReActAgent
from config.settings import ExperimentConfig
from evaluation.metrics import evaluate_correctness, detect_fault_flagged
from experiment.fault_injection import TrialSpec
from tools import TOOL_REGISTRY


CSV_FIELDNAMES = [
    "trial_id",
    "timestamp",
    "model",
    "test_case_id",
    "question",
    "tool",
    "tool_input",
    "ground_truth",
    "difficulty",
    "fault_type",
    "prompt_condition",
    "tool_result",
    "agent_flagged",
    "agent_answer",
    "is_correct",
    "latency_ms",
    "raw_response",
]


def _load_completed_trial_ids(csv_path: Path) -> set[int]:
    """Load trial IDs from an existing results CSV (for resuming)."""
    if not csv_path.exists():
        return set()
    ids: set[int] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ids.add(int(row["trial_id"]))
            except (KeyError, ValueError):
                pass
    return ids


class ExperimentRunner:
    """Executes experiment trials and logs results incrementally."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.llm = LLMClient(config.llm)
        self.agent = ReActAgent(self.llm)

    def run_all(self, trials: list[TrialSpec], output_path: Path | None = None) -> list[dict]:
        """Execute all trials and write results to CSV (incrementally).

        If the output CSV already exists, resumes from where it left off.
        """
        if output_path is None:
            output_path = self.config.output_csv

        # Check for existing progress
        completed_ids = _load_completed_trial_ids(output_path)
        all_results: list[dict] = []

        # Load existing results if resuming
        if completed_ids:
            with open(output_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_results = list(reader)
            print(f"Resuming: {len(completed_ids)} trials already completed")

        total = len(trials)
        remaining = [t for t in trials if t.trial_id not in completed_ids]

        print(f"Running {len(remaining)} remaining trials (of {total}) -> {output_path}")
        print(f"Model: {self.config.llm.model}")
        print(f"Seed: {self.config.seed}")
        print()

        # Open CSV in append mode for incremental writing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_path.exists() and output_path.stat().st_size > 0

        with open(output_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if not file_exists:
                writer.writeheader()

            for i, trial in enumerate(remaining, 1):
                result = self._run_single(trial, trial_id=trial.trial_id)
                all_results.append(result)

                # Write immediately so progress is never lost
                writer.writerow(result)
                f.flush()

                # Progress indicator
                status = "FLAGGED" if result["agent_flagged"] else "ok"
                correct = "CORRECT" if result["is_correct"] else "WRONG"
                done = len(completed_ids) + i
                print(
                    f"  [{done:3d}/{total}] trial={trial.trial_id:3d} "
                    f"id={trial.test_case_id:2d} "
                    f"fault={str(trial.fault_type):13s} "
                    f"verify={trial.prompt_condition} "
                    f"-> {status:7s} {correct:7s} "
                    f"({result['latency_ms']:.0f}ms)"
                )

                # Small delay between trials to avoid rate limiting
                time.sleep(1)

        print(f"\nResults written to {output_path}")
        return all_results

    def _run_single(self, trial: TrialSpec, trial_id: int) -> dict:
        """Execute a single trial and return the result dict."""
        tool_fn = TOOL_REGISTRY[trial.tool]

        # Execute tool with fault injection
        try:
            tool_result = tool_fn(trial.tool_input, fault=trial.fault_type)
            tool_errored = False
        except Exception as e:
            tool_result = f"[TOOL RAISED EXCEPTION: {e}]"
            tool_errored = True

        # Run agent
        start_time = time.perf_counter()
        agent_output = self.agent.run(
            question=trial.question,
            tool_name=trial.tool,
            tool_input=trial.tool_input,
            tool_result=tool_result,
            use_verify_prompt=trial.prompt_condition,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Evaluate
        is_correct = evaluate_correctness(
            agent_answer=agent_output["answer"],
            ground_truth=trial.ground_truth,
            tool=trial.tool,
        )
        agent_flagged = detect_fault_flagged(
            agent_output["raw_response"],
            tool_result,
            fault_type=trial.fault_type,
        )

        return {
            "trial_id": trial_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.config.llm.model,
            "test_case_id": trial.test_case_id,
            "question": trial.question,
            "tool": trial.tool,
            "tool_input": trial.tool_input,
            "ground_truth": trial.ground_truth,
            "difficulty": trial.difficulty,
            "fault_type": trial.fault_type or "none",
            "prompt_condition": "verify" if trial.prompt_condition else "no_verify",
            "tool_result": tool_result[:500],
            "agent_flagged": agent_flagged,
            "agent_answer": agent_output["answer"][:500],
            "is_correct": is_correct,
            "latency_ms": round(latency_ms, 1),
            "raw_response": agent_output["raw_response"][:1000],
        }
