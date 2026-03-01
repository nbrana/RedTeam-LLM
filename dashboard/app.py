import json
import os
from pathlib import Path

from flask import Flask, render_template

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = Path(os.getenv("REDTEAM_RESULTS_FILE", PROJECT_ROOT / "results.json"))


@app.route("/")
def index():
    results = []
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open(encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                pass

    # Calculate stats
    total = len(results)
    safe = len([r for r in results if r.get("safety_label") == "safe"])
    unsafe = len([r for r in results if r.get("safety_label") == "unsafe"])
    borderline = total - safe - unsafe

    stats = {
        "total": total,
        "safe": safe,
        "unsafe": unsafe,
        "borderline": borderline,
    }
    return render_template("index.html", results=results, stats=stats)


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=debug, port=port)
