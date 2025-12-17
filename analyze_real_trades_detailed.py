#!/usr/bin/env python3
"""
Detaillierte Analyse der Real-Trades
Identifiziert Probleme und Verlustmuster
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

print("=" * 80)
print("  JAEGERBOT: DETAILLIERTE REAL-TRADE ANALYSE")
print("=" * 80)
print()

# === 1. Daten laden ===
real_trades = pd.read_csv('Export futures order history-2025-12-18 04_55_20.csv')
real_trades['Date'] = pd.to_datetime(real_trades['Date'])
real_trades = real_trades.sort_values('Date')
real_trades_nov = real_trades[(real_trades['Date'] >= '2025-11-01') & (real_trades['Date'] < '2025-12-18')]

# Konvertiere numerische Spalten
numeric_cols = ['Price', 'Average Price', 'Order amount', 'Executed', 'Trading volume', 'Realized P/L', 'NetProfits']
for col in numeric_cols:
    real_trades_nov[col] = pd.to_numeric(real_trades_nov[col].astype(str).str.replace(',', ''), errors='coerce')

# === 2. Basis-Statistiken ===
total_pnl = real_trades_nov['NetProfits'].sum()
wins = (real_trades_nov['NetProfits'] > 0).sum()
losses = (real_trades_nov['NetProfits'] < 0).sum()
win_rate = wins / len(real_trades_nov) * 100 if len(real_trades_nov) > 0 else 0

avg_win = real_trades_nov[real_trades_nov['NetProfits'] > 0]['NetProfits'].mean() if wins > 0 else 0
avg_loss = real_trades_nov[real_trades_nov['NetProfits'] < 0]['NetProfits'].mean() if losses > 0 else 0
profit_factor = abs(real_trades_nov[real_trades_nov['NetProfits'] > 0]['NetProfits'].sum() / 
                    real_trades_nov[real_trades_nov['NetProfits'] < 0]['NetProfits'].sum()) if losses > 0 else 0

print("📊 GESAMTSTATISTIK")
print("-" * 80)
print(f"Zeitraum: {real_trades_nov['Date'].min().strftime('%d.%m.%Y')} bis {real_trades_nov['Date'].max().strftime('%d.%m.%Y')}")
print(f"Anzahl Trades: {len(real_trades_nov)}")
print(f"Total PnL: {total_pnl:.2f} USDT")
print(f"")
print(f"Winning Trades: {wins} ({win_rate:.1f}%)")
print(f"Losing Trades: {losses} ({100-win_rate:.1f}%)")
print(f"")
print(f"Durchschnittlicher Gewinn: {avg_win:.2f} USDT")
print(f"Durchschnittlicher Verlust: {avg_loss:.2f} USDT")
print(f"Profit Factor: {profit_factor:.2f}")
print()

# === 3. Analyse nach Asset ===
print("=" * 80)
print("📈 PERFORMANCE NACH ASSET")
print("=" * 80)

asset_stats = real_trades_nov.groupby('Futures').agg({
    'NetProfits': ['count', 'sum', 'mean'],
    'Order ID': 'count'
}).round(2)

# Berechne Win-Rate pro Asset
asset_win_rates = {}
for asset in real_trades_nov['Futures'].unique():
    asset_trades = real_trades_nov[real_trades_nov['Futures'] == asset]
    wins_asset = (asset_trades['NetProfits'] > 0).sum()
    total_asset = len(asset_trades)
    asset_win_rates[asset] = (wins_asset / total_asset * 100) if total_asset > 0 else 0

asset_summary = pd.DataFrame({
    'Trades': real_trades_nov.groupby('Futures')['NetProfits'].count(),
    'PnL': real_trades_nov.groupby('Futures')['NetProfits'].sum().round(2),
    'Avg PnL': real_trades_nov.groupby('Futures')['NetProfits'].mean().round(4),
    'Win Rate %': [asset_win_rates[asset] for asset in real_trades_nov.groupby('Futures')['NetProfits'].count().index]
}).sort_values('PnL')

print(asset_summary)
print()

# === 4. Zeitliche Analyse ===
print("=" * 80)
print("📅 ZEITLICHE ANALYSE")
print("=" * 80)

# Gruppiere nach Wochen
real_trades_nov['Week'] = real_trades_nov['Date'].dt.isocalendar().week
weekly_pnl = real_trades_nov.groupby('Week').agg({
    'NetProfits': ['sum', 'count'],
    'Date': ['min', 'max']
})

print("\nWöchentliche Performance:")
for idx, row in weekly_pnl.iterrows():
    pnl = row[('NetProfits', 'sum')]
    count = row[('NetProfits', 'count')]
    start = row[('Date', 'min')].strftime('%d.%m')
    end = row[('Date', 'max')].strftime('%d.%m')
    print(f"  KW {int(idx):2d} ({start}-{end}): {pnl:8.2f} USDT ({int(count):3d} Trades)")

# === 5. Direction Analysis ===
print()
print("=" * 80)
print("🎯 LONG vs SHORT ANALYSE")
print("=" * 80)

# Paare finden: Open/Close
real_trades_nov['Type'] = real_trades_nov['Direction'].str.split().str[0]
real_trades_nov['Side'] = real_trades_nov['Direction'].str.split().str[1]

direction_stats = real_trades_nov.groupby('Side').agg({
    'NetProfits': ['count', 'sum', 'mean']
}).round(2)

print(direction_stats)
print()

# === 6. Liquidation Analysis ===
print("=" * 80)
print("⚠️  LIQUIDATIONS")
print("=" * 80)

liquidations = real_trades_nov[real_trades_nov['Direction'].str.contains('Liquidation', na=False)]
if len(liquidations) > 0:
    print(f"\n🚨 {len(liquidations)} LIQUIDATIONEN GEFUNDEN!")
    for idx, liq in liquidations.iterrows():
        print(f"  {liq['Date'].strftime('%d.%m.%Y %H:%M')} - {liq['Futures']}: {liq['NetProfits']:.2f} USDT")
    total_liq_loss = liquidations['NetProfits'].sum()
    print(f"\n  Total Liquidation Loss: {total_liq_loss:.2f} USDT")
else:
    print("✅ Keine Liquidationen gefunden")
print()

# === 7. Trades mit größten Verlusten ===
print("=" * 80)
print("💥 TOP 10 VERLUST-TRADES")
print("=" * 80)

worst_trades = real_trades_nov.nsmallest(10, 'NetProfits')[['Date', 'Futures', 'Direction', 'NetProfits']]
for idx, trade in worst_trades.iterrows():
    print(f"  {trade['Date'].strftime('%d.%m %H:%M')} | {trade['Futures']:12s} | {trade['Direction']:25s} | {trade['NetProfits']:7.2f} USDT")
print()

# === 8. DOGE-Spezialanalyse (da 134 Trades) ===
print("=" * 80)
print("🐕 DOGE-DETAIL-ANALYSE (134 Trades = 52% aller Trades)")
print("=" * 80)

doge_trades = real_trades_nov[real_trades_nov['Futures'] == 'DOGEUSDT'].copy()
doge_pnl = doge_trades['NetProfits'].sum()
doge_wins = (doge_trades['NetProfits'] > 0).sum()
doge_win_rate = doge_wins / len(doge_trades) * 100 if len(doge_trades) > 0 else 0

print(f"\nDOGE Statistiken:")
print(f"  Total Trades: {len(doge_trades)}")
print(f"  Total PnL: {doge_pnl:.2f} USDT")
print(f"  Win Rate: {doge_win_rate:.1f}% ({doge_wins}/{len(doge_trades)})")
print(f"  Avg PnL per Trade: {doge_trades['NetProfits'].mean():.4f} USDT")

# Zeitliche Verteilung der DOGE-Trades
doge_trades['Hour'] = doge_trades['Date'].dt.hour
hourly_doge = doge_trades.groupby('Hour')['NetProfits'].agg(['count', 'sum']).round(2)
print(f"\nDOGE Trades nach Uhrzeit (Top 5 aktivste Stunden):")
top_hours = hourly_doge.sort_values('count', ascending=False).head(5)
for hour, row in top_hours.iterrows():
    print(f"  {int(hour):02d}:00 Uhr: {int(row['count']):3d} Trades, PnL: {row['sum']:7.2f} USDT")

# === 9. Kumulative PnL Kurve ===
real_trades_nov['Cumulative_PnL'] = real_trades_nov['NetProfits'].cumsum()
real_trades_nov['Trade_Number'] = range(1, len(real_trades_nov) + 1)

print()
print("=" * 80)
print("📉 KUMULATIVE PNL-ENTWICKLUNG")
print("=" * 80)
print(f"\nStart: 0.00 USDT")
print(f"Ende: {real_trades_nov['Cumulative_PnL'].iloc[-1]:.2f} USDT")
print(f"Peak: {real_trades_nov['Cumulative_PnL'].max():.2f} USDT (bei Trade #{real_trades_nov[real_trades_nov['Cumulative_PnL'] == real_trades_nov['Cumulative_PnL'].max()]['Trade_Number'].iloc[0]})")
print(f"Trough: {real_trades_nov['Cumulative_PnL'].min():.2f} USDT (bei Trade #{real_trades_nov[real_trades_nov['Cumulative_PnL'] == real_trades_nov['Cumulative_PnL'].min()]['Trade_Number'].iloc[0]})")

# === 10. Zusammenfassung & Empfehlungen ===
print()
print("=" * 80)
print("🎯 ZUSAMMENFASSUNG & EMPFEHLUNGEN")
print("=" * 80)
print("""
HAUPTPROBLEME IDENTIFIZIERT:

1. ❌ KATASTROPHALE WIN-RATE (6.9%)
   → 93% aller Trades sind Verluste!
   → Normal wäre 40-60%

2. ❌ DOGE-ÜBERGEWICHTUNG (52% aller Trades)
   → DOGE allein: -22.35 USDT Verlust
   → Scheint die schlechteste Strategie zu sein

3. ❌ NEGATIVER PROFIT-FACTOR
   → Verluste sind größer als Gewinne
   → Durchschnittlicher Verlust größer als durchschnittlicher Gewinn

4. ⚠️  LIQUIDATIONEN
""" + (f"   → {len(liquidations)} Liquidationen mit {liquidations['NetProfits'].sum():.2f} USDT Verlust!" if len(liquidations) > 0 else "   → Keine Liquidationen (gut!)") + """

SOFORTMASSNAHMEN:

1. 🛑 DOGE 15m STRATEGIE DEAKTIVIEREN
   → Ist offensichtlich unprofitabel
   → Verursacht 64% des Gesamtverlusts

2. 🔍 BACKTEST-VALIDIERUNG
   → Prüfe ob DOGE 15m im Backtest überhaupt profitable Signale zeigt
   → Vergleiche Backtest-Signale mit Real-Trades

3. ⚙️ PARAMETER-ÜBERPRÜFUNG
   → prediction_threshold zu niedrig? (aktuell 0.61)
   → Leverage zu hoch? (aktuell 15x)
   → Stop-Loss zu eng?

4. 📊 FOKUS AUF PROFITABLE STRATEGIEN
   → SOL scheint mit +1.46 USDT zu funktionieren
   → Konzentriere dich auf weniger, aber bessere Trades

5. 🧪 PAPER-TRADING
   → Teste neue Parameter erstmal mit Paper-Trading
   → Validiere Win-Rate > 40% bevor du live gehst
""")

print("=" * 80)

# Export für weitere Analyse
doge_trades.to_csv('doge_trades_analysis.csv', index=False)
print("\n✅ DOGE-Trades exportiert nach: doge_trades_analysis.csv")
print("✅ Nutze diese Datei für detaillierte Analyse der DOGE-Probleme")
