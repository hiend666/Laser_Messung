# build_pdf.ps1 - Erzeugt docs/Anleitung.pdf aus docs/Anleitung.md
#
# Aufruf:
#   .\docs\build_pdf.ps1
#
# Setzt das venv .\.venv voraus (enthält reportlab + Pillow).

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Split-Path -Parent $here
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py (Join-Path $here "md_to_pdf.py")
