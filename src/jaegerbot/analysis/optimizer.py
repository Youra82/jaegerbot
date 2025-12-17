# src/jaegerbot/analysis/optimizer.py
import os
import sys
import json
import optuna
import numpy as np
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.analysis.backtester import load_data, run_ann_backtest
from jaegerbot.utils.telegram import send_message
from jaegerbot.analysis.evaluator import evaluate_dataset

optuna.logging.set_verbosity(optuna.logging.WARNING)
HISTORICAL_DATA = None
CURRENT_MODEL_PATHS = {}
CURRENT_TIMEFRAME = None
FIXED_THRESHOLD = None

MAX_DRAWDOWN_CONSTRAINT = 0.30
MIN_WIN_RATE_CONSTRAINT = 55.0
MIN_PNL_CONSTRAINT = 0.0
START_CAPITAL = 1000
OPTIM_MODE = "strict"

def objective(trial, symbol):
    # --- KORRIGIERT: Entferne initial_sl_pct, füge ATR-basierte Parameter hinzu ---
    params = {
        'prediction_threshold': FIXED_THRESHOLD,
        'risk_reward_ratio': trial.suggest_float('risk_reward_ratio', 1.0, 5.0),
        'risk_per_trade_pct': trial.suggest_float('risk_per_trade_pct', 0.5, 2.0),
        'leverage': trial.suggest_int('leverage', 5, 25),
        # Neue Parameter für dynamischen SL
        'atr_multiplier_sl': trial.suggest_float('atr_multiplier_sl', 1.0, 4.0),
        'min_sl_pct': trial.suggest_float('min_sl_pct', 0.3, 2.0),
        # Trailing Stop Parameter
        'trailing_stop_activation_rr': trial.suggest_float('trailing_stop_activation_rr', 1.0, 4.0),
        'trailing_stop_callback_rate_pct': trial.suggest_float('trailing_stop_callback_rate_pct', 0.5, 3.0)
    }
    # --- ENDE KORRIGIERT ---

    result = run_ann_backtest(
        HISTORICAL_DATA.copy(),
        params,
        CURRENT_MODEL_PATHS,
        START_CAPITAL,
        timeframe=CURRENT_TIMEFRAME
    )

    pnl, drawdown, trades, win_rate = result.get('total_pnl_pct', -1000), result.get('max_drawdown_pct', 1.0), result.get('trades_count', 0), result.get('win_rate', 0)

    if OPTIM_MODE == "strict" and (drawdown > MAX_DRAWDOWN_CONSTRAINT or win_rate < MIN_WIN_RATE_CONSTRAINT or pnl < MIN_PNL_CONSTRAINT or trades < 50):
        raise optuna.exceptions.TrialPruned()
    elif OPTIM_MODE == "best_profit" and (drawdown > MAX_DRAWDOWN_CONSTRAINT or trades < 50):
        raise optuna.exceptions.TrialPruned()

    drawdown_safe = max(drawdown, 0.01)
    return pnl / drawdown_safe

def create_safe_filename(symbol, timeframe):
    return f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"


def main():
    global HISTORICAL_DATA, CURRENT_MODEL_PATHS, CURRENT_TIMEFRAME, FIXED_THRESHOLD, MAX_DRAWDOWN_CONSTRAINT, MIN_WIN_RATE_CONSTRAINT, MIN_PNL_CONSTRAINT, START_CAPITAL, OPTIM_MODE

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

    for task in TASKS:
        symbol, timeframe = task['symbol'], task['timeframe']
        CURRENT_TIMEFRAME = timeframe

        print(f"\n===== Optimiere: {symbol} ({timeframe}) mit festem Threshold: {FIXED_THRESHOLD} =====")

        safe_filename = create_safe_filename(symbol, timeframe)
        CURRENT_MODEL_PATHS = {'model': os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5'), 'scaler': os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')}

        HISTORICAL_DATA = load_data(symbol, timeframe, args.start_date, args.end_date)
        if HISTORICAL_DATA.empty: continue

        print("\n--- Bewertung der Datensatz-Qualität ---")
        evaluation = evaluate_dataset(HISTORICAL_DATA.copy(), timeframe)
        print(f"Note: {evaluation['score']} / 10\n" + "\n".join(evaluation['justification']) + "\n----------------------------------------")

        DB_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'optuna_studies.db')
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

        STORAGE_URL = f"sqlite:///{DB_FILE}?timeout=60"
        study_name = f"ann_{safe_filename}_{OPTIM_MODE}"

        study = optuna.create_study(storage=STORAGE_URL, study_name=study_name, direction="maximize", load_if_exists=True)

        objective_wrapper = lambda trial: objective(trial, symbol)
        study.optimize(objective_wrapper, n_trials=N_TRIALS, n_jobs=args.jobs, show_progress_bar=True)

        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not valid_trials: print(f"\n❌ FEHLER: Für {symbol} ({timeframe}) konnte keine Konfiguration gefunden werden."); continue

        best_trial = max(valid_trials, key=lambda t: t.value)
        best_params = best_trial.params
        best_params['prediction_threshold'] = FIXED_THRESHOLD

        config_dir = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs')
        os.makedirs(config_dir, exist_ok=True)
        config_output_path = os.path.join(config_dir, f'config_{safe_filename}.json')

        behavior_config = {"use_longs": True, "use_shorts": True}

        # --- KORRIGIERT: Speichere ATR-basierte Parameter (anstelle des alten initial_sl_pct) ---
        config_output = {
            "market": {"symbol": symbol, "timeframe": timeframe},
            "strategy": {"prediction_threshold": FIXED_THRESHOLD},
            "risk": {
                "margin_mode": "isolated",
                "risk_per_trade_pct": round(best_params['risk_per_trade_pct'], 2),
                # Entfernt: "initial_sl_pct"
                "risk_reward_ratio": round(best_params['risk_reward_ratio'], 2),
                "leverage": best_params['leverage'],
                "trailing_stop_activation_rr": round(best_params['trailing_stop_activation_rr'], 2),
                "trailing_stop_callback_rate_pct": round(best_params['trailing_stop_callback_rate_pct'], 2),
                # Neue Parameter
                'atr_multiplier_sl': round(best_params['atr_multiplier_sl'], 2),
                'min_sl_pct': round(best_params['min_sl_pct'], 2)
            },
            "behavior": behavior_config
        }
        # --- ENDE KORRIGIERT ---
        with open(config_output_path, 'w') as f: json.dump(config_output, f, indent=4)
        print(f"\n✔ Beste Konfiguration wurde in '{config_output_path}' gespeichert.")

    try:
        with open(os.path.join(PROJECT_ROOT, 'secret.json'), "r") as f: secrets = json.load(f)
        telegram_config = secrets.get('telegram', {})
    except Exception:
        pass


if __name__ == "__main__":
    main()
