"""Run after editing data/results.json: python sync_data.py
Regenerates data/results.js, the copy used when index.html is opened by double-click."""
import json
d = json.load(open("data/results.json", encoding="utf-8"))
open("data/results.js", "w", encoding="utf-8").write("window.MUON_RESULTS = " + json.dumps(d, ensure_ascii=False) + ";\n")
print("data/results.js updated")
