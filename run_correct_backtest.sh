#!/bin/bash
# Automatischer Backtest mit den ECHTEN Live-Trading-Strategien
# Vergleicht die Performance mit den Real-Trades

echo "============================================================"
echo "  JaegerBot: Korrekter Backtest mit Live-Strategien"
echo "============================================================"
echo ""
echo "Starte Backtest mit den aktiven Strategien aus settings.json:"
echo "  1) config_SOLUSDTUSDT_15m.json"
echo "  18) config_DOGEUSDTUSDT_15m.json"
echo "  23) config_ETHUSDTUSDT_15m.json"
echo "  36) config_XRPUSDTUSDT_15m.json"
echo "  5) config_ADAUSDTUSDT_1d.json"
echo "  1) config_AAVEUSDTUSDT_1d.json"
echo ""
echo "Zeitraum: 2025-11-01 bis 2025-12-17"
echo "Startkapital: 250 USDT (geschätzt aus Real-Trades)"
echo "============================================================"
echo ""

# Führe Show-Results mit automatischer Eingabe aus
{
  echo "2"           # Modus 2: Manuelle Portfolio-Simulation
  echo "2025-11-01"  # Startdatum
  echo ""            # Enddatum (Standard = Heute)
  echo "250"         # Startkapital
  echo "30,18,23,36,5,1"  # Strategie-Nummern
} | bash show_results.sh

echo ""
echo "============================================================"
echo "  Backtest abgeschlossen!"
echo "============================================================"
echo ""
echo "Starte nun Vergleich mit Real-Trades..."
C:/Users/matol/Desktop/bots/jaegerbot/.venv/Scripts/python.exe compare_real_vs_backtest.py
