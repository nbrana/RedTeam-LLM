# Defensive G2PIA Module

This repository includes a defensive, **lab-only** implementation of Goal-guided Generative Prompt Injection Attacks (G2PIA). The code mirrors the methodology from the referenced paper to measure robustness; it does not ship a deployable attack surface.

## Safety checklist
- Default run mode is clean-only. Candidate injections are produced **only** when `--allow-adversarial-sim` is explicitly passed.
- Generator prompts include explicit safety banners and neutral phrasing.
- Candidate generation falls back to deterministic templates when no local LLM is available; results stay in local logs.
- Reports clearly state the lab-only intent and include the safety banner.

## Pipeline
1. **Core word extraction:** spaCy (if available) or a heuristic extracts subject/predicate/object.
2. **Core pool creation:** WordNet synonyms + numeric variants produce alternative predicate/object pairs.
3. **Generation prompt:** A neutral template instructs a local Ollama model to return a benign sentence with one number.
4. **Candidate filtering:** Accept only if semantic distance `D(t', t) < ε` (word-level embeddings) **and** `|cos(w(t'), w(t)) − γ| < δ` (sentence embeddings).
5. **Evaluation loop:** Insert the candidate into the clean prompt, call the victim model, log Clean Accuracy, Attack Accuracy, ASR, and confusion buckets.
6. **Reporting:** JSON + optional HTML outputs with per-sample detail and safety banner.

Default hyperparameters (configurable in `config.yaml`):
```
epsilon = 0.2
gamma = 0.5
delta = 0.05
N = 50
max_attempts = 500
```

## Quickstart (offline-friendly)
```bash
# Clean-only metrics (no adversarial simulation)
redteam run-g2pia --dataset tests/data/g2pia_synthetic.jsonl \
  --victim-model ollama/gpt-local \
  --out reports/g2pia_sample.json

# Full simulation (lab only)
redteam run-g2pia --dataset tests/data/g2pia_synthetic.jsonl \
  --victim-model ollama/gpt-local \
  --candidates 5 \
  --allow-adversarial-sim \
  --out reports/g2pia_lab_run.json \
  --html-out reports/g2pia_lab_run.html
```

## Reproducing sample results
- Sample dataset: `tests/data/g2pia_synthetic.jsonl` (10 arithmetic questions).
- Sample report: `reports/sample_g2pia_report.json` created with the offline stubbed victim model for demonstration.
- Tests: `pytest tests/test_g2pia.py` (uses stubbed embeddings and victim to avoid network access).

## Extending locally
- Swap the generator/victim model names in `config.yaml` (`g2pia` block).
- Provide custom embedding functions to `evaluate_attack_set` if you have local SentenceTransformer/word2vec checkpoints.
- For larger datasets (GSM8K/SQuAD), ensure they are stored locally in JSON/JSONL with `question/input` and `answer/ground_truth` fields.
