#!/bin/bash
# Offline-Tests (keine echten Orders): Regeln, Datenpruefung, Live-Zyklus gegen Schein-Boerse, Paritaet
cd "$(dirname "$0")"
PY=.venv/bin/python3; [ -x "$PY" ] || PY=$(command -v python3 || command -v python)
export PYTHONPATH=src
$PY -m pytest -q tests && $PY tests/parity_check.py --candles 60
