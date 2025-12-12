# Windows Setup Guide (Local, Offline-Friendly)

This project is designed to run locally with Ollama-backed models. The steps below assume Windows 10/11.

## Prerequisites
- **Python 3.10+** (use the official installer from python.org; check “Add Python to PATH”).
- **Git** (optional but recommended).
- **Ollama for Windows** (required for LLM calls):
  1. Download and install Ollama from https://ollama.com/download/windows.
  2. Start the Ollama service (it runs as a background service after install).
  3. Pull at least one model, e.g.:
     ```powershell
     ollama pull llama3
     ollama pull phi3:mini
     ```

## Clone and install dependencies
```powershell
git clone https://github.com/<your-org>/RedTeam-LLM.git
cd RedTeam-LLM
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .  # install the package and CLI entrypoint locally
```

> If you prefer `uv` on Windows:
> ```powershell
> pip install uv
> uv venv
> .venv\Scripts\activate
> uv pip install -r requirements.txt
> uv pip install -e .
> ```

## Verify Ollama connectivity
```powershell
ollama run phi3:mini "Say hello"
```
If you get a response, the service is reachable.

## Run the core red team loop
```powershell
python -m redteam_llm.cli run --config config.yaml --output results.json
```
or, if installed via entrypoint:
```powershell
redteam run --config config.yaml --output results.json
```

## Run the defensive G2PIA harness (lab-only)
1. Make sure your dataset is local JSON/JSONL (e.g., `tests\data\g2pia_synthetic.jsonl`).
2. Run clean-only metrics (no adversarial sampling):
   ```powershell
   redteam run-g2pia --dataset tests/data/g2pia_synthetic.jsonl `
     --victim-model phi3:mini `
     --out reports/g2pia_clean.json
   ```
3. Enable adversarial candidate generation (lab mode):
   ```powershell
   redteam run-g2pia --dataset tests/data/g2pia_synthetic.jsonl `
     --victim-model phi3:mini `
     --candidates 5 `
     --allow-adversarial-sim `
     --out reports/g2pia_lab_run.json `
     --html-out reports/g2pia_lab_run.html
   ```

## Notes and troubleshooting
- **Model pulls**: Large models may take time; ensure enough disk space.
- **Ports**: Ollama uses its own local service port; if blocked by security software, allow local connections.
- **GPU**: Ollama will use GPU if supported; otherwise falls back to CPU.
- **Missing deps**: If `spacy` model download fails, run:
  ```powershell
  python -m spacy download en_core_web_sm
  ```
- **Offline runs**: The G2PIA tests and sample run can fall back to deterministic stubs when no embeddings/models are present, but real generation requires the pulled Ollama models.
