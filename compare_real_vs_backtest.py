#!/usr/bin/env python3
"""
Vergleich: Real-Trades vs Backtest
Analysiert die Abweichungen zwischen tatsächlichen Bitget-Trades und Backtest-Ergebnissen
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# === 1. Real-Trades laden ===
print("=" * 60)
print("Lade Real-Trades von Bitget...")
print("=" * 60)

real_trades = pd.read_csv('Export futures order history-2025-12-18 04_55_20.csv')
real_trades['Date'] = pd.to_datetime(real_trades['Date'])
real_trades = real_trades.sort_values('Date')

# Filter: nur November 2025
real_trades_nov = real_trades[(real_trades['Date'] >= '2025-11-01') & (real_trades['Date'] < '2025-12-18')]

print(f"\n📊 REAL-TRADES (1. Nov - 17. Dez 2025)")
print(f"   Anzahl Trades: {len(real_trades_nov)}")
print(f"   Zeitraum: {real_trades_nov['Date'].min()} bis {real_trades_nov['Date'].max()}")

# Berechne kumulative PnL
real_trades_nov['Cumulative_PnL'] = real_trades_nov['NetProfits'].astype(float).cumsum()
total_pnl_real = real_trades_nov['NetProfits'].astype(float).sum()

print(f"   Total PnL: {total_pnl_real:.2f} USDT")
print(f"   Startkapital (geschätzt): ~250 USDT")
print(f"   Performance: {(total_pnl_real / 250 * 100):.2f}%")

# Trades nach Asset
trades_by_asset = real_trades_nov.groupby('Futures').agg({
    'NetProfits': ['count', 'sum']
}).round(2)
trades_by_asset.columns = ['Anzahl', 'PnL']
print("\n   Trades nach Asset:")
print(trades_by_asset.sort_values('Anzahl', ascending=False))

# Win Rate
wins = (real_trades_nov['NetProfits'].astype(float) > 0).sum()
total = len(real_trades_nov)
win_rate = wins / total * 100
print(f"\n   Win Rate: {win_rate:.1f}% ({wins}/{total})")

# === 2. Backtest laden ===
print("\n" + "=" * 60)
print("Lade Backtest-Daten...")
print("=" * 60)

backtest = pd.read_csv('manual_portfolio_equity.csv')
backtest['timestamp'] = pd.to_datetime(backtest['timestamp'])

# Filter November
backtest_nov = backtest[(backtest['timestamp'] >= '2025-11-01') & (backtest['timestamp'] < '2025-12-18')]

print(f"\n📈 BACKTEST (Portfolio-Simulation)")
print(f"   Datenpunkte: {len(backtest_nov)}")
print(f"   Zeitraum: {backtest_nov['timestamp'].min()} bis {backtest_nov['timestamp'].max()}")

if len(backtest_nov) > 0:
    start_equity = backtest_nov['equity'].iloc[0]
    end_equity = backtest_nov['equity'].iloc[-1]
    backtest_pnl = end_equity - start_equity
    backtest_return = (end_equity / start_equity - 1) * 100
    max_dd = backtest_nov['drawdown_pct'].max() * 100
    
    print(f"   Start Equity: {start_equity:.2f} USDT")
    print(f"   End Equity: {end_equity:.2f} USDT")
    print(f"   Total PnL: {backtest_pnl:.2f} USDT")
    print(f"   Performance: {backtest_return:.2f}%")
    print(f"   Max Drawdown: {max_dd:.2f}%")
else:
    print("   ⚠️  WARNUNG: Keine Daten für November-Dezember gefunden!")
    print(f"   Verfügbare Daten: {backtest['timestamp'].min()} bis {backtest['timestamp'].max()}")

# === 3. Vergleich ===
print("\n" + "=" * 60)
print("🔍 VERGLEICH: Real vs Backtest")
print("=" * 60)

if len(backtest_nov) > 0:
    print(f"\n   Real-Trades:")
    print(f"   - Anzahl Trades: {len(real_trades_nov)}")
    print(f"   - PnL: {total_pnl_real:.2f} USDT")
    print(f"   - Return: {(total_pnl_real / 250 * 100):.2f}%")
    print(f"   - Win Rate: {win_rate:.1f}%")
    
    print(f"\n   Backtest:")
    print(f"   - Anzahl Trades: 2 (laut Terminal-Output)")
    print(f"   - PnL: {backtest_pnl:.2f} USDT")
    print(f"   - Return: {backtest_return:.2f}%")
    print(f"   - Max DD: {max_dd:.2f}%")
    
    print(f"\n   ⚠️  MASSIVE DISKREPANZ:")
    print(f"   - Real: {len(real_trades_nov)} Trades mit {total_pnl_real:.2f} USDT PnL")
    print(f"   - Backtest: 2 Trades mit {backtest_pnl:.2f} USDT PnL")
    print(f"   - Faktor Trades: {len(real_trades_nov) / 2:.0f}x mehr in der Realität!")

# === 4. Trades-Matching ===
print("\n" + "=" * 60)
print("🔎 TRADE-MATCHING: Welche Assets wurden gehandelt?")
print("=" * 60)

# Extrahiere Basis-Asset
real_trades_nov['BaseAsset'] = real_trades_nov['Futures'].str.replace('USDT', '')
real_assets = set(real_trades_nov['BaseAsset'].unique())
print(f"\n   Real-Trades Assets: {sorted(real_assets)}")
print(f"   Backtest Assets: ['ADA', 'AAVE'] (laut Terminal)")

backtest_assets = {'ADA', 'AAVE'}
overlap = real_assets & backtest_assets
print(f"   Überschneidung: {sorted(overlap)}")
print(f"   Nur Real: {sorted(real_assets - backtest_assets)}")

# === 5. Zusammenfassung ===
print("\n" + "=" * 60)
print("📌 ZUSAMMENFASSUNG")
print("=" * 60)
print("""
HAUPTPROBLEME:
1. ❌ Trade-Frequenz: Real 262 Trades vs Backtest 2 Trades
2. ❌ Asset-Auswahl: Real traded hauptsächlich DOGE, Backtest nur ADA+AAVE
3. ❌ Performance-Vergleich schwierig durch unterschiedliche Strategien
4. ❌ Backtest-Daten enden am 11. November (nicht 17. Dezember!)

MÖGLICHE URSACHEN:
- Bot in der Realität nutzt andere/mehr Strategien als im Backtest
- Höhere Zeitauflösung (15m) wird in Realität genutzt, aber nicht im Backtest
- Unterschiedliches Kapital-Management
- Backtest läuft mit falschen/fehlenden Modellen

EMPFEHLUNG:
1. Backtest mit allen aktiven Strategien wiederholen
2. Startkapital an Realität anpassen
3. Zeitraum prüfen (warum endet Backtest am 11.11.?)
4. Config-Files der Real-Trades identifizieren
""")

print("=" * 60)
