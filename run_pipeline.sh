#!/bin/bash
# Komplette Pipeline von Hand (sonst automatisch zu Quartalsbeginn ueber den master_runner):
# Sicherung -> Kerzen -> neues Quartalsmodell -> Regeln -> Portfolio -> Paritaetstest; bei Fehler Ruecksicherung.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python3; [ -x "$PY" ] || PY=$(command -v python3 || command -v python)
export PYTHONPATH=src
echo "JaegerBot v5 Pipeline -- Fortschritt: logs/auto_optimizer.log, Details: logs/pipeline_*.log"
$PY auto_optimizer_scheduler.py --force
echo
tail -n 12 logs/auto_optimizer.log
echo
./show_results.sh
