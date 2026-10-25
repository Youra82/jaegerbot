# src/jaegerbot/analysis/walkforward.py
"""Walk-Forward: je Quartalsbeginn ein Ensemble trainieren (nur Vergangenheit) und jede Kerze
mit dem Modell ihres Quartals vorhersagen.

Aufruf: python -m jaegerbot.analysis.walkforward --timeframes 4h [--first-cutoff 2023-01-01] [--jobs 6] [--only-new]
Ergebnis: artifacts/models/<tf>/ensemble_<cutoff>.joblib und data/predictions/<tf>/<COIN>.csv
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.data import load_data, symbol_of, load_settings
from jaegerbot.utils.data_check import check_candles
from jaegerbot.strategy.features import compute_features, FEATURES, WARMUP_CANDLES
from jaegerbot.strategy import model as mdl
from jaegerbot.strategy import trade_rules as tr

PRED_DIR = tr.profile_dir(os.path.join(PROJECT_ROOT, 'data', 'predictions'))
FEAT_DIR = tr.profile_dir(os.path.join(PROJECT_ROOT, 'data', 'features'))
MAX_TRAIN_ROWS = 400_000
logger = logging.getLogger(__name__)


def coin_frame(coin, timeframe):
    """Features + Labels einer Coin; Zeilen erst nach dem Einschwingen."""
    raw = load_data(symbol_of(coin), timeframe)
    problems = check_candles(raw, timeframe, min_candles=WARMUP_CANDLES + 200,
                             max_age=pd.Timedelta(days=2) + pd.Timedelta(timeframe))
    if problems:
        print(f"  {coin} {timeframe}: ausgeschlossen -- {'; '.join(problems)}", flush=True)
        return None
    f = compute_features(raw)
    lab = mdl.labels(f, timeframe)
    idx_ns = f.index.asi8
    for side in ('L', 'S'):
        ex = lab[f'exit{side}'].to_numpy()
        # Zeitpunkt (ns) der Ausstiegskerze; -1 = Ausgang noch offen
        lab[f'exit_ns{side}'] = np.where(ex >= 0, idx_ns[np.clip(ex, 0, None)], -1)
    f = pd.concat([f, lab], axis=1).iloc[WARMUP_CANDLES:]
    f = f.dropna(subset=FEATURES + ['atr'])
    f['coin'] = coin
    return f


def build_dataset(coins, timeframe):
    os.makedirs(os.path.join(FEAT_DIR, timeframe), exist_ok=True)
    frames = []
    for coin in coins:
        f = coin_frame(coin, timeframe)
        if f is None:
            continue
        f.to_pickle(os.path.join(FEAT_DIR, timeframe, f"{coin}.pkl"))
        # Gesamtdatensatz schlank halten (1h: ~2,5 Mio. Zeilen) -- nur Trainings-/Vorhersagespalten, float32
        keep = f[FEATURES + ['labL', 'labS', 'exit_nsL', 'exit_nsS', 'coin']].copy()
        keep[FEATURES] = keep[FEATURES].astype(np.float32)
        frames.append(keep)
    return pd.concat(frames)


def refresh_predictions(coin, timeframe):
    """Features + Vorhersagen einer Coin mit den vorhandenen Modellen neu schreiben (ohne Training) --
    fuer den taeglichen Selbstcheck, damit der Backtest die juengsten Kerzen kennt."""
    f = coin_frame(coin, timeframe)
    if f is None:
        return False
    os.makedirs(os.path.join(FEAT_DIR, timeframe), exist_ok=True)
    f.to_pickle(os.path.join(FEAT_DIR, timeframe, f"{coin}.pkl"))
    cutoffs = mdl.available_cutoffs(timeframe)
    parts = []
    for k, c in enumerate(cutoffs):
        nxt = cutoffs[k + 1] if k + 1 < len(cutoffs) else None
        sub = f[(f.index >= c) & ((f.index < nxt) if nxt is not None else True)]
        if len(sub):
            parts.append(mdl.predict_frame(mdl.load(timeframe, c), sub))
    if not parts:
        return False
    os.makedirs(os.path.join(PRED_DIR, timeframe), exist_ok=True)
    pd.concat(parts).to_csv(os.path.join(PRED_DIR, timeframe, f"{coin}.csv"))
    return True


def _train_one(timeframe, cutoff, X, labL, labS, exitL, exitS, seeds):
    cutoff = pd.Timestamp(cutoff)
    c64 = cutoff.value
    # Purge: nur Kerzen, deren Ausstieg (Label-Ausgang) vor dem cutoff lag
    keep = (exitL < c64) & (exitL > 0) & (exitS < c64) & (exitS > 0)
    rows = pd.DataFrame(X[keep], columns=FEATURES)
    rows['labL'], rows['labS'] = labL[keep], labS[keep]
    if len(rows) > MAX_TRAIN_ROWS:
        rows = rows.sample(MAX_TRAIN_ROWS, random_state=0)
    ens = mdl.train_ensemble(rows, timeframe, cutoff, seeds)
    mdl.save(ens)
    return f"{cutoff:%Y-%m-%d}: {len(rows)} Trainingskerzen, L {ens.n_train['L']} / S {ens.n_train['S']} Labels"


def main():
    rs = load_settings()['research']
    p = argparse.ArgumentParser()
    p.add_argument('--timeframes', nargs='*', default=rs['timeframes'])
    p.add_argument('--coins', nargs='*', default=rs['universe'])
    p.add_argument('--first-cutoff', default='2023-01-01')
    p.add_argument('--jobs', type=int, default=6)
    p.add_argument('--only-new', action='store_true', help='vorhandene Modelle nicht neu trainieren')
    a = p.parse_args()
    seeds = tuple(range(int(rs.get('ensemble_seeds', 3))))

    for tf in a.timeframes:
        print(f"\n===== {tf}: Features + Labels =====", flush=True)
        data = build_dataset(a.coins, tf)
        X = data[FEATURES].to_numpy(dtype=np.float32)
        labL, labS = data['labL'].to_numpy(), data['labS'].to_numpy()
        exitL, exitS = data['exit_nsL'].to_numpy(), data['exit_nsS'].to_numpy()

        cutoffs = mdl.quarter_starts(a.first_cutoff, pd.Timestamp.now(tz='UTC'))
        have = set(mdl.available_cutoffs(tf)) if a.only_new else set()
        todo = [c for c in cutoffs if c not in have]
        print(f"{tf}: {len(data)} Kerzen aus {data['coin'].nunique()} Coins, {len(todo)} Modelle zu trainieren", flush=True)
        msgs = Parallel(n_jobs=a.jobs, verbose=0)(
            delayed(_train_one)(tf, c, X, labL, labS, exitL, exitS, seeds) for c in todo)
        for m in msgs:
            print(f"  {tf} {m}", flush=True)

        # Vorhersagen: jede Kerze mit dem Modell ihres Quartals
        os.makedirs(os.path.join(PRED_DIR, tf), exist_ok=True)
        cutoffs = mdl.available_cutoffs(tf)
        ens = {c: mdl.load(tf, c) for c in cutoffs}
        for coin, g in data.groupby('coin'):
            parts = []
            for k, c in enumerate(cutoffs):
                nxt = cutoffs[k + 1] if k + 1 < len(cutoffs) else None
                sub = g[(g.index >= c) & ((g.index < nxt) if nxt is not None else True)]
                if len(sub):
                    parts.append(mdl.predict_frame(ens[c], sub))
            if parts:
                pd.concat(parts).to_csv(os.path.join(PRED_DIR, tf, f"{coin}.csv"))
        print(f"{tf}: Vorhersagen geschrieben ({data['coin'].nunique()} Coins)", flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    main()
