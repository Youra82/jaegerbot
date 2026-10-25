#!/bin/bash
# Offline-Tests (keine echten Orders): Regeln, Datenpruefung, Live-Zyklus gegen Schein-Boerse, Paritaet
cd "$(dirname "$0")"
PY=.venv/bin/python3; [ -x "$PY" ] || PY=$(command -v python3 || command -v python)
export PYTHONPATH=src
# Long-Profil (trend) und Short-Profil (mensch) getrennt -- das Profil gilt je Prozess
$PY -m pytest -q tests && $PY tests/parity_check.py --candles 60 && JAEGER_PROFILE=mensch $PY -m pytest -q tests/test_trade_cycle.py && JAEGER_PROFILE=mensch $PY tests/parity_check.py --candles 20
