#!/bin/bash
# Dieses Skript führt das komplette Test-Sicherheitsnetz aus.
echo "--- Starte JaegerBot-Sicherheitsnetz ---"

# Stell sicher, dass beim Live-Test PEPE mit kleinem Notional getestet wird
# Setzt das Notional (USDT) für PEPE auf 2 USDT und übergibt den Margin-Mode an die Tests
export JAEGER_PEPE_NOTIONAL_USDT=${JAEGER_PEPE_NOTIONAL_USDT:-2}
export JAEGER_MARGIN_MODE=${JAEGER_MARGIN_MODE:-isolated}

echo "-> Test-Umgebungsvariablen: JAEGER_PEPE_NOTIONAL_USDT=${JAEGER_PEPE_NOTIONAL_USDT}, JAEGER_MARGIN_MODE=${JAEGER_MARGIN_MODE}"

# Aktiviere die virtuelle Umgebung
source .venv/bin/activate

# Führe pytest aus. -v für mehr Details, -s um print() Ausgaben anzuzeigen.
python3 -m pytest -v -s

# Deaktiviere die Umgebung wieder
deactivate

echo "--- Sicherheitscheck abgeschlossen ---"
