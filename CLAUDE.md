# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for analyzing CSV data from a Laservibrometer (laser measurement device). Displays displacement, velocity, and acceleration curves and exports results as PDF/PNG.

## Running the App

```bash
streamlit run app.py
```

Or via the helper script (activates venv first):
```bash
./run.sh
```

The app runs on port 8501. In the dev container it starts with `--server.enableCORS false --server.enableXsrfProtection false`.

## Install Dependencies

```bash
pip install -r requirements.txt
```

Python 3.11 required. Key packages: streamlit, pandas, plotly, kaleido, scipy, reportlab.

## Architecture

Four source files — no build steps, no tests:

| File | Lines | Inhalt |
|---|---|---|
| `app.py` | ~2250 | Streamlit UI, Session State, Callbacks, Hauptfluss |
| `chart.py` | ~1010 | Plotly-Chart-Aufbau, PNG-Export (Kaleido), PDF-Export (ReportLab), Y-Achsen-Layout |
| `reader.py` | ~645 | CSV-Parsing, SG-Filter-Ableitung |
| `pages/1_Datei_Merger.py` | ~470 | Multipage-Seite: Kanäle aus mehreren Dateien zu einer CSV/TXT/Oszilloskop-CSV zusammenführen |

**Data flow:**
1. User uploads a CSV file in the sidebar
2. `load_rohdaten` (cached) parses it via `reader.load_raw`
3. Y-Offsets and X-Offsets are applied from session state
4. Velocity and acceleration are derived via Savitzky-Golay filter (`reader.berechne_sg_ableitung`)
5. `compute_best_fit_rectangle` detects rectangular pulse shapes for hub analysis
6. `chart._finde_sop_kreuzungen` finds the Speed-on-Point crossing on the rising edge
7. `chart._yachsen_layout` assigns Plotly y-axes (merges same-unit channels, splits by SPLIT_FAKTOR or manual limits)
8. Plotly renders an interactive multi-axis chart (displacement / velocity / acceleration)
9. Metrics are displayed in a 3-row card grid (Zeit/Weg, Geschwindigkeit, Beschleunigung)
10. `chart.build_chart_png` renders PNG via Kaleido; `chart.build_pdf` builds A4-landscape PDF via ReportLab

**Key constants in `chart.py`** (imported into app.py):
- `KANAL_FARBEN`, `FARBE_*` — all diagram colors
- `SPLIT_FAKTOR = 15.0` — threshold for splitting same-unit channels onto separate axes
- `_ZEIT_TO_S` — unit-to-seconds conversion dict

## Streamlit Session State Pattern

To avoid widget feedback loops, the app uses a **two-key pattern** (documented near the top of the main block in app.py):

- **Free keys** (`xa`, `xb`, `off1`, `off2`, …): hold canonical values, updated by callbacks
- **Widget keys** (`xa_sw`, `xa_nw`, `xb_sw`, …): bound to sliders/number inputs; their `on_change` callbacks write back to the free keys

Always maintain this separation when adding new interactive controls. The defaults dict (near the top of app.py) must be kept in sync with any new session state keys.

## UI Language

All user-facing text, variable names, and code comments are in **German**. Keep this consistent when adding UI elements or comments.

## Version

Tracked as `VERSION = "v1.05.00"` at the top of `app.py`. Update this string when making notable changes.
