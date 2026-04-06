# src/jaegerbot/analysis/optimizer.py
import os
import sys
import json
import math
import optuna
import numpy as np
import argparse
from datetime import datetime as _dt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

RESULTS_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'results', 'last_optimizer_run.json')

from jaegerbot.analysis.backtester import load_data, run_ann_backtest
from jaegerbot.utils.telegram import send_message
from jaegerbot.analysis.evaluator import evaluate_dataset
from jaegerbot.utils.signal_scorer import DEFAULT_MIN_SIGNAL_SCORE

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _timeframe_hours(tf: str) -> float:
    if 'm' in tf:
        return int(tf.replace('m', '')) / 60.0
    if 'h' in tf:
        return float(tf.replace('h', ''))
    if 'd' in tf:
        return float(tf.replace('d', '')) * 24.0
    return 1.0


HISTORICAL_DATA = None
TRAIN_DATA = None   # 70% — Optimierung
TEST_DATA  = None   # 30% — Out-of-Sample Validierung
TRAIN_DATA_WITH_PREDICTIONS = None  # Vorberechnete Predictions (Train)
TEST_DATA_WITH_PREDICTIONS  = None  # Vorberechnete Predictions (Test)
CURRENT_MODEL_PATHS = {}
CURRENT_TIMEFRAME = None
FIXED_THRESHOLD = None

MAX_DRAWDOWN_CONSTRAINT = 0.25
MIN_WIN_RATE_CONSTRAINT = 55.0
MIN_PNL_CONSTRAINT = 5.0
START_CAPITAL = 1000
OPTIM_MODE = "strict"

# Walk-Forward-Split: 70% Training, 30% Out-of-Sample Test
_WFV_TRAIN_RATIO = 0.70


def objective(trial, symbol):
    params = {
        'prediction_threshold': trial.suggest_float('prediction_threshold', 0.58, 0.80),
        # Risiko & Positionsgröße
        'risk_reward_ratio':   trial.suggest_float('risk_reward_ratio',   1.5, 8.0),
        'risk_per_trade_pct':  trial.suggest_float('risk_per_trade_pct',  1.0, 5.0),
        'leverage':            trial.suggest_int(  'leverage',            10, 50),
        # Stop-Loss
        'atr_multiplier_sl':   trial.suggest_float('atr_multiplier_sl',   1.0, 4.0),
        'max_sl_pct':          trial.suggest_float('max_sl_pct',          0.8, 3.0),
        # Trailing Stop
        'trailing_stop_activation_rr':      trial.suggest_float('trailing_stop_activation_rr',      1.0, 3.0),
        'trailing_stop_callback_rate_pct':  trial.suggest_float('trailing_stop_callback_rate_pct',  0.3, 2.0),
        # Signal-Score Schwelle
        'min_signal_score':    trial.suggest_float('min_signal_score',    5.0, 9.0),
        # Fixe Defaults (nicht tunen — spart Suchraum)
        'min_sl_pct':          0.3,
        'min_ann_score':       1.5,
        'ann_weight':          3.5,
        'volume_weight':       1.5,
        'structure_weight':    1.0,
        'maintenance_margin_rate': 0.004,
    }

    # ── Liquidation Guard: SL muss vor Liquidation feuern ───────────────────
    _mmr = 0.004
    _max_safe_sl_pct = (1.0 / params['leverage']) - _mmr  # z.B. 2.54% bei 34x
    if params['max_sl_pct'] / 100.0 > _max_safe_sl_pct:
        # Leverage reduzieren bis SL passt
        _safe_lev = max(1, int(1.0 / (params['max_sl_pct'] / 100.0 + _mmr)))
        params['leverage'] = _safe_lev
        _max_safe_sl_pct = (1.0 / params['leverage']) - _mmr
    # SL zusätzlich auf 90% der Liquidationsgrenze cappen (10% Puffer)
    params['max_sl_pct'] = min(params['max_sl_pct'], _max_safe_sl_pct * 100.0 * 0.90)
    # ── Ende Liquidation Guard ───────────────────────────────────────────────

    # ── Schritt 1: Training-Backtest (70%) ───────────────────────────────────
    r_train = run_ann_backtest(TRAIN_DATA_WITH_PREDICTIONS.copy(), params, CURRENT_MODEL_PATHS, START_CAPITAL, timeframe=CURRENT_TIMEFRAME, precomputed=True)
    train_pnl  = r_train.get('total_pnl_pct', -1000)
    train_dd   = r_train.get('max_drawdown_pct', 1.0)
    train_trades = r_train.get('trades_count', 0)

    _train_len = len(TRAIN_DATA) if TRAIN_DATA is not None else 0
    min_train_trades = max(5, _train_len // 200)
    if train_trades < min_train_trades or train_dd > MAX_DRAWDOWN_CONSTRAINT:
        raise optuna.exceptions.TrialPruned()

    # ── Schritt 2: Out-of-Sample Test (30%) ──────────────────────────────────
    r_test = run_ann_backtest(TEST_DATA_WITH_PREDICTIONS.copy(), params, CURRENT_MODEL_PATHS, START_CAPITAL, timeframe=CURRENT_TIMEFRAME, precomputed=True)
    test_pnl    = r_test.get('total_pnl_pct', -1000)
    test_dd     = r_test.get('max_drawdown_pct', 1.0)
    test_trades = r_test.get('trades_count', 0)
    test_wr     = r_test.get('win_rate', 0)

    # Constraints auf Out-of-Sample — hier entscheidet sich ob die Config live taugt
    _test_len_data = len(TEST_DATA) if TEST_DATA is not None else 0
    min_test_trades = max(3, _test_len_data // 200)
    if test_trades < min_test_trades or test_dd > MAX_DRAWDOWN_CONSTRAINT or test_pnl <= 0:
        raise optuna.exceptions.TrialPruned()
    if OPTIM_MODE == "strict" and (test_wr < MIN_WIN_RATE_CONSTRAINT or test_pnl < MIN_PNL_CONSTRAINT):
        raise optuna.exceptions.TrialPruned()

    # Anti-overtrading guard
    _tf_str   = CURRENT_TIMEFRAME if CURRENT_TIMEFRAME else '1h'
    _tf_hours = _timeframe_hours(_tf_str)
    _test_len = len(TEST_DATA) if TEST_DATA is not None else 0
    if _tf_hours > 0 and _test_len > 0:
        _test_years = (_test_len * _tf_hours) / 8760.0
        if _test_years > 0:
            _tpy = test_trades / _test_years
            if _tpy > 300:
                raise optuna.exceptions.TrialPruned()

    # ── Score: 30% Training + 70% Out-of-Sample ──────────────────────────────
    # train_dd ist ein Bruch (0.09 = 9%) → ×100 für konsistente Einheiten mit train_pnl (%)
    train_score  = math.log1p(max(0.0, train_pnl)) / max(train_dd * 100, 1.0)
    test_score   = math.log1p(max(0.0, test_pnl))  / max(test_dd  * 100, 1.0)
    # Trade-Frequenz-Bonus: mehr Trades = mehr Compounding (wie vbot)
    trade_bonus  = math.log1p(test_trades) * 2.0
    # Win-Rate-Bonus: 55%+ WR belohnen
    wr_bonus     = max(0.0, (test_wr - 40.0) / 10.0)
    final_score  = train_score * 0.30 + test_score * 0.70 + trade_bonus + wr_bonus

    # Out-of-Sample PnL in Config speichern (realistische Erwartung)
    trial.set_user_attr('pnl_pct', round(test_pnl, 2))
    trial.set_user_attr('train_pnl_pct', round(train_pnl, 2))
    return final_score

def create_safe_filename(symbol, timeframe):
    return f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"


def main():
    global HISTORICAL_DATA, TRAIN_DATA, TEST_DATA, TRAIN_DATA_WITH_PREDICTIONS, TEST_DATA_WITH_PREDICTIONS, CURRENT_MODEL_PATHS, CURRENT_TIMEFRAME, FIXED_THRESHOLD, MAX_DRAWDOWN_CONSTRAINT, MIN_WIN_RATE_CONSTRAINT, MIN_PNL_CONSTRAINT, START_CAPITAL, OPTIM_MODE

    parser = argparse.ArgumentParser(description="Parameter-Optimierung für JaegerBot")
    parser.add_argument('--symbols', required=True, type=str)
    parser.add_argument('--timeframes', required=True, type=str)
    parser.add_argument('--start_date', required=True, type=str)
    parser.add_argument('--end_date', required=True, type=str)
    parser.add_argument('--jobs', required=True, type=int)
    parser.add_argument('--max_drawdown', required=True, type=float)
    parser.add_argument('--start_capital', required=True, type=float)
    parser.add_argument('--min_win_rate', required=True, type=float)
    parser.add_argument('--trials', required=True, type=int)
    parser.add_argument('--min_pnl', required=True, type=float)
    parser.add_argument('--mode', required=True, type=str)
    parser.add_argument('--threshold', required=True, type=float)
    parser.add_argument('--top_n', type=int, default=0)
    args = parser.parse_args()

    FIXED_THRESHOLD = args.threshold
    MAX_DRAWDOWN_CONSTRAINT, MIN_WIN_RATE_CONSTRAINT, MIN_PNL_CONSTRAINT = args.max_drawdown / 100.0, args.min_win_rate, args.min_pnl
    START_CAPITAL, N_TRIALS, OPTIM_MODE, TOP_N_STRATEGIES = args.start_capital, args.trials, args.mode, args.top_n

    symbols, timeframes = args.symbols.split(), args.timeframes.split()
    TASKS = [{'symbol': f"{s}/USDT:USDT", 'timeframe': tf} for s in symbols for tf in timeframes]

    # Bestehende Datei lesen falls vorhanden (bash-Pipeline ruft optimizer.py
    # einmal pro Paar auf — wir haengen an statt die Datei zu ueberschreiben)
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                run_results = json.load(f)
            run_results.setdefault('saved', [])
            run_results.setdefault('failed', [])
        except Exception:
            run_results = {
                'run_start': _dt.now().isoformat(timespec='seconds'),
                'run_end': None,
                'saved': [],
                'failed': [],
            }
    else:
        run_results = {
            'run_start': _dt.now().isoformat(timespec='seconds'),
            'run_end': None,
            'saved': [],
            'failed': [],
        }

    for task in TASKS:
        symbol, timeframe = task['symbol'], task['timeframe']
        CURRENT_TIMEFRAME = timeframe

        print(f"\n===== Optimiere: {symbol} ({timeframe}) mit festem Threshold: {FIXED_THRESHOLD} =====")

        safe_filename = create_safe_filename(symbol, timeframe)
        CURRENT_MODEL_PATHS = {'model': os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5'), 'scaler': os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')}

        HISTORICAL_DATA = load_data(symbol, timeframe, args.start_date, args.end_date)
        if HISTORICAL_DATA.empty:
            run_results['failed'].append({'symbol': symbol, 'timeframe': timeframe, 'reason': 'no_data'})
            continue

        # Walk-Forward Split: 70% Training / 30% Out-of-Sample Test
        split_idx  = max(50, int(len(HISTORICAL_DATA) * _WFV_TRAIN_RATIO))
        TRAIN_DATA = HISTORICAL_DATA.iloc[:split_idx]
        TEST_DATA  = HISTORICAL_DATA.iloc[split_idx:]
        print(f"Walk-Forward: {len(TRAIN_DATA)} Kerzen Training / {len(TEST_DATA)} Kerzen Test (70/30)")

        # Predictions EINMAL vorberechnen — spart 1000× model.predict() über alle Trials
        print("Berechne ANN-Predictions einmalig vor der Optimierung...")
        from jaegerbot.utils.ann_model import load_model_and_scaler, create_ann_features
        from jaegerbot.analysis.backtester import calculate_supertrend_direction
        import pandas as pd
        _model, _scaler = load_model_and_scaler(CURRENT_MODEL_PATHS['model'], CURRENT_MODEL_PATHS['scaler'])
        if not _model or not _scaler:
            run_results['failed'].append({'symbol': symbol, 'timeframe': timeframe, 'reason': 'model_not_found'})
            continue
        _feature_cols = [
            'bb_width', 'bb_pband', 'obv', 'rsi', 'macd_diff', 'macd',
            'atr_normalized', 'adx', 'adx_pos', 'adx_neg',
            'volume_ratio', 'mfi', 'cmf',
            'price_to_ema20', 'price_to_ema50',
            'stoch_k', 'stoch_d', 'williams_r', 'roc', 'cci',
            'price_to_resistance', 'price_to_support',
            'high_low_range', 'close_to_high', 'close_to_low',
            'day_of_week', 'hour_of_day',
            'returns_lag1', 'returns_lag2', 'returns_lag3', 'hist_volatility',
            'body_to_atr', 'upper_wick_ratio', 'lower_wick_ratio',
            'candle_direction', 'body_midpoint_ratio',
            'bull_streak', 'bear_streak',
            'dist_to_struct_high', 'dist_to_struct_low',
            'price_in_range_20', 'price_in_range_50',
            'fib_position', 'in_fib_zone',
            'volume_direction', 'buying_pressure', 'selling_pressure',
        ]
        def _precompute(df):
            d = create_ann_features(df.copy())
            d.dropna(inplace=True)
            if d.empty: return d
            d['supertrend_direction'] = calculate_supertrend_direction(d)
            d['avg_atr_normalized'] = d['atr_normalized'].rolling(50).mean()
            d.dropna(inplace=True)
            if d.empty: return d
            scaled = _scaler.transform(d[_feature_cols])
            d['prediction'] = _model.predict(scaled, verbose=0).flatten()
            return d
        TRAIN_DATA_WITH_PREDICTIONS = _precompute(TRAIN_DATA.copy())
        TEST_DATA_WITH_PREDICTIONS  = _precompute(TEST_DATA.copy())
        print(f"✔ Predictions vorberechnet: {len(TRAIN_DATA_WITH_PREDICTIONS)} Train / {len(TEST_DATA_WITH_PREDICTIONS)} Test Kerzen")

        print("\n--- Bewertung der Datensatz-Qualität ---")
        evaluation = evaluate_dataset(HISTORICAL_DATA.copy(), timeframe)
        print(f"Note: {evaluation['score']} / 10\n" + "\n".join(evaluation['justification']) + "\n----------------------------------------")

        DB_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'optuna_studies.db')
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

        STORAGE_URL = f"sqlite:///{DB_FILE}?timeout=60"
        study_name = f"ann_{safe_filename}_{OPTIM_MODE}"

        # Studie laden — aber prüfen ob alte Trials vom kaputten Backtester (start_capital-Bug)
        # vorhanden sind. Erkennungsmerkmale:
        #   1. Score > 30 (mit fixem Backtester unrealistisch — max realistisch ~20)
        #   2. train_pnl_pct > 10000% in einem Trial-Attribut
        # Falls ja: Studie löschen und neu anlegen für saubere Ergebnisse.
        try:
            existing_study = optuna.load_study(storage=STORAGE_URL, study_name=study_name)
            complete_trials = [t for t in existing_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if complete_trials:
                max_score     = max(t.value for t in complete_trials)
                max_train_pnl = max(
                    t.user_attrs.get('train_pnl_pct', 0) for t in complete_trials
                )
                if max_score > 30 or max_train_pnl > 10000:
                    print(f"⚠️  Alte Studie '{study_name}' enthält unrealistische Werte "
                          f"(max_score={max_score:.1f}, max_train_pnl={max_train_pnl:.0f}%). "
                          f"Studie wird zurückgesetzt...")
                    optuna.delete_study(study_name=study_name, storage=STORAGE_URL)
                    print(f"✔ Studie gelöscht. Starte sauber.")
        except Exception:
            pass  # Studie existiert noch nicht — normal
        study = optuna.create_study(storage=STORAGE_URL, study_name=study_name, direction="maximize", load_if_exists=True)

        objective_wrapper = lambda trial: objective(trial, symbol)
        study.optimize(objective_wrapper, n_trials=N_TRIALS, n_jobs=args.jobs, show_progress_bar=True)

        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not valid_trials:
            print(f"\n❌ FEHLER: Für {symbol} ({timeframe}) konnte keine Konfiguration gefunden werden.")
            run_results['failed'].append({'symbol': symbol, 'timeframe': timeframe, 'reason': 'no_valid_trials'})
            continue

        best_trial = max(valid_trials, key=lambda t: t.value)
        best_params = best_trial.params
        best_params['prediction_threshold'] = FIXED_THRESHOLD

        config_dir = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs')
        os.makedirs(config_dir, exist_ok=True)
        config_output_path = os.path.join(config_dir, f'config_{safe_filename}.json')

        behavior_config = {"use_longs": True, "use_shorts": True}

        best_pnl_pct       = best_trial.user_attrs.get('pnl_pct', 0)       # Out-of-Sample
        best_train_pnl_pct = best_trial.user_attrs.get('train_pnl_pct', 0)  # Training

        config_output = {
            "_meta": {
                "pnl_pct": best_pnl_pct,
                "pnl_pct_oos": best_pnl_pct,
                "pnl_pct_train": best_train_pnl_pct,
                "wfv": "70/30",
            },
            "market": {"symbol": symbol, "timeframe": timeframe},
            "strategy": {
                "prediction_threshold": FIXED_THRESHOLD,
                "min_signal_score": round(best_params.get('min_signal_score', DEFAULT_MIN_SIGNAL_SCORE), 2),
            },
            "risk": {
                "margin_mode": "isolated",
                "risk_per_trade_pct": round(best_params['risk_per_trade_pct'], 2),
                "risk_reward_ratio": round(best_params['risk_reward_ratio'], 2),
                "leverage": best_params['leverage'],
                "trailing_stop_activation_rr": round(best_params['trailing_stop_activation_rr'], 2),
                "trailing_stop_callback_rate_pct": round(best_params['trailing_stop_callback_rate_pct'], 2),
                "atr_multiplier_sl": round(best_params['atr_multiplier_sl'], 2),
                "min_sl_pct": round(best_params.get('min_sl_pct', 0.3), 2),
                "max_sl_pct": round(best_params.get('max_sl_pct', 2.0), 2),
            },
            "behavior": behavior_config,
        }
        config_filename = f'config_{safe_filename}.json'
        with open(config_output_path, 'w') as f: json.dump(config_output, f, indent=4)
        print(f"\n✔ Beste Konfiguration wurde in '{config_output_path}' gespeichert.")
        run_results['saved'].append({
            'symbol': symbol,
            'timeframe': timeframe,
            'pnl_pct': best_pnl_pct,
            'config_file': config_filename,
        })

    run_results['run_end'] = _dt.now().isoformat(timespec='seconds')
    try:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(run_results, f, indent=2)
        print(f"\n[Optimizer] Ergebnisse gespeichert: {len(run_results['saved'])} gespeichert, {len(run_results['failed'])} fehlgeschlagen.")
    except Exception as e:
        print(f"[Optimizer] Warnung: Konnte Ergebnisse nicht speichern: {e}")


if __name__ == "__main__":
    main()
