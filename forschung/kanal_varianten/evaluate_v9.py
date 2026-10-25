"""V9 (saubere Linien) gegen V7 = Live-Profil 'mensch' -- Auswahl nur im IS, dann einmal OOS (PREREG.md, Abschnitt V9).

  python forschung/kanal_varianten/evaluate_v9.py
"""
import os
import sys
import pickle
import subprocess
import importlib.util
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
TMP = os.environ.get('TMP', '/tmp')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    P = _load('p', os.path.join(ROOT, 'forschung', 'pairs', 'evaluate.py'))
    from jaegerbot.strategy import trade_rules as tr
    ev_script = os.path.join(ROOT, 'forschung', 'kanal_varianten', 'evaluate.py')
    for prof in ('mensch', 'kv9'):
        env = {**os.environ, 'JAEGER_PROFILE': prof, 'PYTHONPATH': os.path.join(ROOT, 'src')}
        subprocess.run([sys.executable, ev_script, '--child', prof], env=env, cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    trend = pickle.load(open(P.TREND_EVENTS, 'rb'))
    long_ev = []
    for e in trend['events']:
        if e['exit_time'] is None:
            continue
        ef = tr.fill_price(e['signal_close'], e['side'], True)
        pnl, _ = tr.trade_pnl(e['side'], 1.0, ef, e['exit_raw'])
        long_ev.append({'time': e['time'], 'exit_time': e['exit_time'], 'coins': (e['coin'],),
                        'R': pnl / abs(ef - e['sl']), 'risk': 0.95})

    def is_best(ev):
        sub = [e for e in ev if e['time'] < P.OOS]
        bb = None
        for sc in np.round(np.arange(0.05, 3.01, 0.05), 2):
            st = P.simulate(sub, None, scale=float(sc))
            if st['dd'] <= 30 and (bb is None or st['end'] > bb[1]['end']):
                bb = (float(sc), st)
        return bb
    res = {}
    for prof in ('mensch', 'kv9'):
        d = pickle.load(open(os.path.join(TMP, f'kv_{prof}.pkl'), 'rb'))
        bb = is_best(long_ev + d['events'])
        res[prof] = (d, bb)
        c = d['cfg']
        print(f"{prof:7s} Regel {c['sides']} Top {100 - c['quantile'] * 100:.1f} % Trend={int(c['ema_filter'])} | "
              f"{len(d['events'])} Short-Signale | IS Kombi mit Long: {P.cagr(bb[1], P.START, P.OOS):+.1f} %/J DD {bb[1]['dd']:.1f} (Faktor {bb[0]})")
    better = res['kv9'][1][1]['end'] > res['mensch'][1][1]['end']
    print('IS-Auswahl:', 'V9 (saubere Linien)' if better else 'V7 bleibt (V9 im IS nicht besser)')
    if better:
        for prof in ('kv9', 'mensch'):
            d, bb = res[prof]
            o = P.simulate([e for e in long_ev + d['events'] if e['time'] >= P.OOS], None, scale=bb[0])
            res[prof] = (d, bb, o)
            print(f"OOS 26 W einmalig ({prof}, IS-Faktor {bb[0]}): {o['pnl']:+.1f} % DD {o['dd']:.1f} n {o['n']}")
        ok = res['kv9'][2]['end'] >= res['mensch'][2]['end'] and res['kv9'][2]['dd'] <= 30
        print('Ergebnis:', 'V9 bestaetigt -> ersetzt die Short-Seite' if ok else 'V9 im OOS nicht bestaetigt -> V7 bleibt')


if __name__ == '__main__':
    main()
