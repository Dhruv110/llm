#!/usr/bin/env bash
# Serve gpu_compare.html and gpu_specs.csv so the page can load the CSV.
# Run from the llm folder, then open: http://localhost:9000/gpu_compare.html

cd "$(dirname "$0")"
echo "Serving from: $(pwd)"
echo "Open in browser: http://localhost:9000/gpu_compare.html"
echo ""
python3 -m http.server 9000
