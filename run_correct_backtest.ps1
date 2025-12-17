# PowerShell Script: Korrekter Backtest mit Live-Strategien

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  JaegerBot: Korrekter Backtest mit Live-Strategien" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starte Backtest mit den aktiven Strategien aus settings.json:"
Write-Host "  30) config_SOLUSDTUSDT_15m.json"
Write-Host "  18) config_DOGEUSDTUSDT_15m.json"
Write-Host "  23) config_ETHUSDTUSDT_15m.json"
Write-Host "  36) config_XRPUSDTUSDT_15m.json"
Write-Host "  5) config_ADAUSDTUSDT_1d.json"
Write-Host "  1) config_AAVEUSDTUSDT_1d.json"
Write-Host ""
Write-Host "Zeitraum: 2025-11-01 bis 2025-12-17"
Write-Host "Startkapital: 250 USDT (geschätzt aus Real-Trades)"
Write-Host "============================================================"
Write-Host ""

# Erstelle Eingabedatei für automatische Eingaben
$input = @"
2
2025-11-01

250
30,18,23,36,5,1
"@

# Führe show_results.sh mit automatischen Eingaben aus
$input | bash show_results.sh

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Backtest abgeschlossen!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Starte nun Vergleich mit Real-Trades..."
& "C:/Users/matol/Desktop/bots/jaegerbot/.venv/Scripts/python.exe" compare_real_vs_backtest.py
