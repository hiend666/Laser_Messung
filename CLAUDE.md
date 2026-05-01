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

All logic lives in a single file: `app.py` (~1000 lines). There are no modules, tests, or build steps.

**Data flow:**
1. User uploads a CSV file in the sidebar
2. `load_csv` (cached) parses it — supports two formats: raw numeric columns or header-based
3. `build_time_axis` constructs the time vector from sample rate settings
4. `apply_offsets` applies Y-axis corrections
5. Velocity and acceleration are derived via Savitzky-Golay filter (scipy)
6. `compute_best_fit_rectangle` detects rectangular pulse shapes for hub analysis
7. Plotly renders an interactive multi-axis chart (displacement / velocity / acceleration)
8. Metrics (Δt, Δs, v-mid, v-max, freq, Δv, hub, a-max) are displayed in a 2×4 card grid
9. PDF export uses ReportLab; PNG export uses Kaleido

## Streamlit Session State Pattern

To avoid widget feedback loops, the app uses a **two-key pattern** (documented at lines ~197–198 of app.py):

- **Free keys** (`xa`, `xb`, `off1`, `off2`, …): hold canonical values, updated by callbacks
- **Widget keys** (`xa_sw`, `xa_nw`, `xb_sw`, …): bound to sliders/number inputs; their `on_change` callbacks write back to the free keys

Always maintain this separation when adding new interactive controls. The defaults dict (lines ~36–63) must be kept in sync with any new session state keys.

## UI Language

All user-facing text, variable names, and code comments are in **German**. Keep this consistent when adding UI elements or comments.

## Version

Tracked as `VERSION = "v1.00.03"` at the top of `app.py`. Update this string when making notable changes.
