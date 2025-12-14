from flask import Flask, render_template
import json
import os

app = Flask(__name__)

RESULTS_FILE = 'results.json'

@app.route('/')
def index():
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                pass
    
    # Calculate stats
    total = len(results)
    safe = len([r for r in results if r.get('safety_label') == 'safe'])
    unsafe = len([r for r in results if r.get('safety_label') == 'unsafe'])
    borderline = total - safe - unsafe
    
    return render_template('index.html', results=results, stats={'total': total, 'safe': safe, 'unsafe': unsafe, 'borderline': borderline})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
