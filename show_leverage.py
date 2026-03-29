#!/usr/bin/env python3
"""Zeigt Hebel und Risikoparameter aller aktiven Strategien aus settings.json."""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

SETTINGS_PATH = os.path.join(PROJECT_ROOT, 'settings.json')
CONFIGS_DIR   = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs')
RESULTS_PATH  = os.path.join(PROJECT_ROOT, 'artifacts', 'results', 'optimization_results.json')

def main():
    try:
        with open(SETTINGS_PATH) as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Fehler: settings.json nicht gefunden.")
        sys.exit(1)

    live = settings.get('live_trading_settings', {})
    use_auto = live.get('use_auto_optimizer_results', False)

    active_files = []

    if use_auto:
        mode_label = "Autopilot (optimization_results.json)"
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH) as f:
                res = json.load(f)
            active_files = res.get('optimal_portfolio', [])
        else:
            print("Fehler: optimization_results.json nicht gefunden.")
            sys.exit(1)
    else:
        mode_label = "Manuell (settings.json)"
        for s in live.get('active_strategies', []):
            if not isinstance(s, dict) or not s.get('active', True):
                continue
            symbol_clean = s['symbol'].replace('/', '').replace(':', '')
            tf = s['timeframe']
            for candidate in [f"config_{symbol_clean}_{tf}.json",
                               f"config_{symbol_clean}_{tf}_macd.json"]:
                if os.path.exists(os.path.join(CONFIGS_DIR, candidate)):
                    active_files.append(candidate)
                    break
            else:
                print(f"  ⚠  Config für {s['symbol']} {tf} nicht gefunden.")

    if not active_files:
        print("Keine aktiven Konfigurationen gefunden.")
        sys.exit(0)

    # Header
    print()
    print(f"  Modus: {mode_label}")
    print(f"  Strategien: {len(active_files)}")
    print()
    header = f"  {'Strategie':<22} {'Hebel':>6}  {'Risiko/Trade':>13}  {'R:R':>5}  {'Max-SL':>7}  {'PnL OOS':>9}  {'Margin':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for filename in active_files:
        if not filename.startswith('config_'):
            filename = f"config_{filename}"
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        full_path = os.path.join(CONFIGS_DIR, filename)
        if not os.path.exists(full_path):
            print(f"  ⚠  {filename} nicht gefunden.")
            continue

        with open(full_path) as f:
            cfg = json.load(f)

        risk  = cfg.get('risk', {})
        meta  = cfg.get('_meta', {})
        mkt   = cfg.get('market', {})

        symbol    = mkt.get('symbol', '').split('/')[0]
        tf        = mkt.get('timeframe', filename.replace('config_', '').replace('.json', ''))
        label     = f"{symbol}/{tf}" if symbol else filename.replace('config_', '').replace('.json', '')

        leverage  = risk.get('leverage', '—')
        risk_pct  = risk.get('risk_per_trade_pct', '—')
        rr        = risk.get('risk_reward_ratio', '—')
        max_sl    = risk.get('max_sl_pct', '—')
        margin    = risk.get('margin_mode', '—')
        pnl_oos   = meta.get('pnl_pct_oos', meta.get('pnl_pct', '—'))

        lev_str  = f"{leverage}x"  if isinstance(leverage, (int, float)) else str(leverage)
        risk_str = f"{risk_pct:.2f}%" if isinstance(risk_pct, (int, float)) else str(risk_pct)
        rr_str   = f"{rr:.2f}"    if isinstance(rr, (int, float)) else str(rr)
        sl_str   = f"{max_sl:.1f}%" if isinstance(max_sl, (int, float)) else str(max_sl)
        pnl_str  = f"{pnl_oos:+.1f}%" if isinstance(pnl_oos, (int, float)) else str(pnl_oos)

        print(f"  {label:<22} {lev_str:>6}  {risk_str:>13}  {rr_str:>5}  {sl_str:>7}  {pnl_str:>9}  {margin:>10}")

    print()

if __name__ == '__main__':
    main()
