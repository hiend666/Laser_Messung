#!/bin/bash

# Startet die Streamlit-App direkt über die virtuelle Umgebung (kein source nötig).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/.venv/bin/streamlit" run "$SCRIPT_DIR/app.py" \
    --server.enableCORS false \
    --server.enableXsrfProtection false
