# LLM Agent Tool-Failure Reliability Benchmark

A reproducible research framework for measuring how reliably LLM agents detect and recover from tool failures.

## Research Question

When a tool silently returns a wrong, malformed, or empty result, how often does the agent:
- Detect the issue and flag it?
- Recover and still produce a correct answer?
- Present a confidently wrong final answer?

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export NVIDIA_API_KEY="nvapi-..."

# 3. Run the experiment
python main.py
```

## Project Structure

```
.
├── main.py                    # CLI entry point
├── config/
│   └── settings.py            # All configuration (LLM, faults, paths, CLI args)
├── tools/
│   ├── calculator.py          # Calculator with fault injection
│   ├── lookup.py              # Wikipedia lookup with fault injection
│   └── converter.py           # Unit converter (14 unit pairs) with fault injection
├── agent/
│   ├── llm_client.py          # Model-agnostic OpenAI-compatible client
│   └── react_agent.py         # ReAct Thought→Action→Observation→Answer loop
├── experiment/
│   ├── test_cases.py          # Load evaluation_dataset.csv
│   ├── fault_injection.py     # Deterministic fault scheduling
│   └── runner.py              # Trial execution and CSV logging
├── evaluation/
│   └── metrics.py             # Correctness checking, metric computation
├── visualization/
│   └── charts.py              # Publication-quality matplotlib charts
├── evaluation_dataset.csv     # 66 test cases (20 calculator + 24 lookup + 22 converter)
├── data/                      # Results CSV written here
├── figures/                   # Charts written here
└── requirements.txt
```

## Configuration

### LLM Provider

The framework uses the OpenAI-compatible chat completions API. Default provider is NVIDIA NIM.

```bash
# NVIDIA NIM (default)
python main.py --model meta/llama-3.1-70b-instruct

# OpenAI
python main.py --base-url https://api.openai.com/v1 --model gpt-4o

# Local (e.g., vLLM, Ollama)
python main.py --base-url http://localhost:8000/v1 --model local-model
```

### Reproducibility

All randomness is seeded:

```bash
python main.py --seed 42    # default
python main.py --seed 123   # different seed, same experimental design
```

### Re-analyze Existing Results

```bash
python main.py --analyze-only data/results.csv
```

## Experiment Design

### Trial Matrix

- **66 test cases** × **5 fault types** × **2 prompt conditions** = **660 trials**
- Fault types: `none`, `silent_wrong`, `error`, `malformed`, `empty`
- Prompt conditions: with/without verification instruction

### Fault Injection

Each tool supports four fault modes:

| Fault | Behavior |
|-------|----------|
| `none` | Correct result |
| `silent_wrong` | Plausible but incorrect result |
| `error` | Tool raises an exception |
| `malformed` | Garbled/nonsensical output |
| `empty` | Empty string response |

### Metrics

| Metric | Definition |
|--------|------------|
| **Silent Failure Rate** | % of `silent_wrong` faults where agent did NOT flag AND answer was wrong |
| **Detection Rate** | % of faulted trials where agent flagged the issue |
| **Recovery Rate** | % of flagged trials where final answer was still correct |
| **Accuracy (Clean)** | % correct on `none` fault trials |
| **Accuracy (Overall)** | % correct across all trials |

## Output

### CSV (`data/results.csv`)

Columns: `trial_id`, `timestamp`, `model`, `test_case_id`, `question`, `tool`, `tool_input`, `ground_truth`, `difficulty`, `fault_type`, `prompt_condition`, `tool_result`, `agent_flagged`, `agent_answer`, `is_correct`, `latency_ms`, `raw_response`

### Charts (`figures/`)

1. `detection_by_fault.png` — Detection rate by fault type (grouped bar)
2. `accuracy_by_condition.png` — Accuracy by prompt condition (grouped bar)
3. `fault_prompt_heatmap.png` — Fault × Prompt condition heatmap
4. `summary_metrics.png` — Summary metrics table

## Swapping Models

The framework is model-agnostic. Any OpenAI-compatible endpoint works:

```bash
# Compare two models
python main.py --model meta/llama-3.1-70b-instruct --seed 42
python main.py --model meta/llama-3.1-8b-instruct --seed 42

# Then compare data/results.csv files
```

## License

Research use only. See DATASET_README.md for dataset construction methodology.
