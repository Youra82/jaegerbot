#!/usr/bin/env python3
"""Zeigt Hebel, SL, TSL-Callback und Risikoparameter aller aktiven Strategien."""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

SETTINGS_PATH = os.path.join(PROJECT_ROOT, 'settings.json')
CONFIGS_DIR   = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs')
RESULTS_PATH  = os.path.join(PROJECT_ROOT, 'artifacts', 'results', 'optimization_results.json')

def fmt(val, suffix='', decimals=2, fallback='—'):
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}{suffix}"
    return fallback

def main():
    try:
        with open(SETTINGS_PATH) as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Fehler: settings.json nicht gefunden.")
        sys.exit(1)

    live     = settings.get('live_trading_settings', {})
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
                print(f"  WARN  Config fuer {s['symbol']} {tf} nicht gefunden.")

    if not active_files:
        print("Keine aktiven Konfigurationen gefunden.")
        sys.exit(0)

    print()
    print(f"  Modus     : {mode_label}")
    print(f"  Strategien: {len(active_files)}")

    for filename in active_files:
        if not filename.startswith('config_'):
            filename = f"config_{filename}"
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        full_path = os.path.join(CONFIGS_DIR, filename)
        if not os.path.exists(full_path):
            print(f"\n  WARN  {filename} nicht gefunden.")
            continue

        with open(full_path) as f:
            cfg = json.load(f)

        risk = cfg.get('risk', {})
        meta = cfg.get('_meta', {})
        mkt  = cfg.get('market', {})

        symbol = mkt.get('symbol', '').split('/')[0]
        tf     = mkt.get('timeframe', '')
        label  = f"{symbol}/{tf}" if symbol else filename.replace('config_', '').replace('.json', '')

        leverage     = risk.get('leverage')
        risk_pct     = risk.get('risk_per_trade_pct')
        rr           = risk.get('risk_reward_ratio')
        min_sl       = risk.get('min_sl_pct')
        max_sl       = risk.get('max_sl_pct')
        atr_mult     = risk.get('atr_multiplier_sl')
        activation   = risk.get('trailing_stop_activation_rr')
        callback     = risk.get('trailing_stop_callback_rate_pct')
        margin       = risk.get('margin_mode', '—')
        pnl_oos      = meta.get('pnl_pct_oos', meta.get('pnl_pct'))

        # SL-Bereich als String
        if isinstance(min_sl, (int, float)) and isinstance(max_sl, (int, float)):
            sl_str = f"{min_sl:.2f}% - {max_sl:.2f}%"
        else:
            sl_str = '—'

        # ATR-Multiplikator
        atr_str = f"ATR x{atr_mult:.2f}" if isinstance(atr_mult, (int, float)) else '—'

        # TSL-Aktivierung relativ zu SL-Distanz
        act_str = f"@ {activation:.2f}x SL-Distanz" if isinstance(activation, (int, float)) else '—'

        # Callback
        cb_str = f"{callback:.2f}%" if isinstance(callback, (int, float)) else 'kein Callback'

        print()
        print(f"  {'=' * 52}")
        print(f"  {label}")
        print(f"  {'=' * 52}")
        print(f"  Hebel          : {fmt(leverage, 'x', 0)}")
        print(f"  Risiko/Trade   : {fmt(risk_pct, '%')}")
        print(f"  Risk-Reward    : {fmt(rr, '', 2)}")
        print(f"  Margin         : {margin}")
        print(f"  ---")
        print(f"  SL             : {sl_str}  ({atr_str})")
        print(f"  TSL Aktivierung: {act_str}")
        print(f"  TSL Callback   : {cb_str}")
        print(f"  ---")
        print(f"  PnL OOS        : {fmt(pnl_oos, '%', 1, 'n/a')}")

    print()

if __name__ == '__main__':
    main()
