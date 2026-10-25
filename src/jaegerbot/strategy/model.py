# src/jaegerbot/strategy/model.py
"""Gepooltes ANN-Ensemble je Timeframe und Seite (Long/Short getrennt).

Ein Modell gehoert zu einem Stichtag (cutoff, Quartalsbeginn): trainiert nur mit Kerzen,
deren Ausgang vor dem cutoff feststand, und sagt die Kerzen ab cutoff bis zum naechsten
Stichtag voraus. Live nimmt das juengste Modell -- dieselbe Datei, die der Backtest fuer
diese Kerzen nimmt.
"""
import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from jaegerbot.strategy.features import FEATURES
from jaegerbot.strategy import trade_rules as tr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = tr.profile_dir(os.path.join(PROJECT_ROOT, 'artifacts', 'models'))


def labels(df, timeframe):
    """Je Kerze und Seite: 1 = TP vor SL, 0 = SL zuerst, NaN = Zeitausstieg/offen.
    Dazu der Index der Ausstiegskerze (fuer den Purge vor dem cutoff)."""
    o, h, l, c, atr = (df[k].to_numpy() for k in ('open', 'high', 'low', 'close', 'atr'))
    n = len(df)
    hold = tr.MAX_HOLD[timeframe]
    out = {}
    for side, key in (('long', 'L'), ('short', 'S')):
        lab = np.full(n, np.nan)
        ex = np.full(n, -1)
        for i in range(n):
            if not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            sl, tp = tr.levels(c[i], atr[i], side)
            r = tr.simulate_exit(o, h, l, c, i, side, sl, tp, hold)
            if r is None:
                continue
            ex[i] = r[0]
            if r[2] != 'time':
                lab[i] = 1.0 if r[2] == 'tp' else 0.0
        out[f'lab{key}'] = lab
        out[f'exit{key}'] = ex
    return pd.DataFrame(out, index=df.index)


class Ensemble:
    """Mittel ueber mehrere MLPs (verschiedene Seeds) je Seite + Schwellen-Quantile."""

    def __init__(self, timeframe, cutoff, members, quantiles, n_train):
        self.timeframe = timeframe
        self.cutoff = pd.Timestamp(cutoff)
        self.members = members        # {'L': [(scaler, mlp), ...], 'S': [...]}
        self.quantiles = quantiles    # {'thrL_950': p, ...}
        self.n_train = n_train

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = {}
        for side, mem in self.members.items():
            out[f'p{side}'] = np.mean([m.predict_proba(s.transform(X))[:, 1] for s, m in mem], axis=0)
        return out


def train_ensemble(train_rows, timeframe, cutoff, seeds=(0, 1, 2)):
    """train_rows: DataFrame mit FEATURES + labL/labS (beliebig viele Coins)."""
    members, quantiles, n_train = {}, {}, {}
    for side in ('L', 'S'):
        d = train_rows.dropna(subset=[f'lab{side}'])
        X = d[FEATURES].to_numpy(dtype=np.float64)
        y = d[f'lab{side}'].astype(int).to_numpy()
        mem = []
        for seed in seeds:
            sc = StandardScaler().fit(X)
            m = MLPClassifier((64, 32), alpha=1e-3, early_stopping=True, max_iter=200, random_state=seed)
            m.fit(sc.transform(X), y)
            mem.append((sc, m))
        members[side] = mem
        # Schwellen auf allen Trainingskerzen (auch Zeitausstiege) -- wie spaeter gehandelt wird
        Xall = train_rows[FEATURES].to_numpy(dtype=np.float64)
        p = np.mean([m.predict_proba(s.transform(Xall))[:, 1] for s, m in mem], axis=0)
        for q in tr.QUANTILE_LEVELS:
            quantiles[tr.thr_col(side, q)] = float(np.quantile(p, q))
        n_train[side] = int(len(d))
    return Ensemble(timeframe, cutoff, members, quantiles, n_train)


def model_path(timeframe, cutoff):
    return os.path.join(MODEL_DIR, timeframe, f"ensemble_{pd.Timestamp(cutoff):%Y-%m-%d}.joblib")


_TRAINING_ONLY = ('_optimizer', '_best_coefs', '_best_intercepts', 'loss_curve_', 'validation_scores_',
                  '_no_improvement_count', 'best_validation_score_')


def strip(ens):
    """Trainingszustand entfernen (Adam-Momente, Lernkurven) -- fuer predict_proba nicht noetig,
    machte die Datei ~4x groesser (Modelle liegen im Git-Repo)."""
    for mem in ens.members.values():
        for _, m in mem:
            for attr in _TRAINING_ONLY:
                if hasattr(m, attr):
                    setattr(m, attr, None)
    return ens


def save(ens):
    path = model_path(ens.timeframe, ens.cutoff)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(strip(ens), path, compress=3)
    return path


def available_cutoffs(timeframe):
    files = sorted(glob.glob(os.path.join(MODEL_DIR, timeframe, 'ensemble_*.joblib')))
    return [pd.Timestamp(os.path.basename(f)[9:19], tz='UTC') for f in files]


_cache = {}


def load(timeframe, cutoff):
    """Mit Prozess-Cache (Schluessel inkl. Datei-Aenderungszeit -- neu trainierte Datei wird neu geladen)."""
    path = model_path(timeframe, cutoff)
    key = (path, os.path.getmtime(path))
    if key not in _cache:
        _cache[key] = joblib.load(path)
    return _cache[key]


def model_for(timeframe, ts):
    """Juengstes Modell mit cutoff <= ts (fuer Live: ts = Signalkerze)."""
    ts = pd.Timestamp(ts)
    cs = [c for c in available_cutoffs(timeframe) if c <= ts]
    return load(timeframe, cs[-1]) if cs else None


def quarter_starts(start, end):
    st = pd.Timestamp(start)
    st = st.tz_convert(None) if st.tzinfo else st
    end = pd.Timestamp(end)
    end = end if end.tzinfo else end.tz_localize('UTC')
    return list(pd.date_range(st.to_period('Q').start_time.tz_localize('UTC'), end, freq='QS'))


def predict_frame(ens, feat_df):
    """Vorhersagen + Schwellen fuer ein Feature-DataFrame (Zeilen ohne NaN in FEATURES)."""
    p = ens.predict(feat_df[FEATURES])
    out = pd.DataFrame(p, index=feat_df.index)
    for k, v in ens.quantiles.items():
        out[k] = v
    out['model'] = f"{ens.cutoff:%Y-%m-%d}"
    return out
