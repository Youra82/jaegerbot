"""Auswertung PREREG.md (Kanal-Varianten). Auswahl nur im IS, danach einmal OOS fuer gewaehlte Variante + B0.

  python forschung/kanal_varianten/evaluate.py geohtf kv1 kv5 kv7 kv8 ...
Je Profil laeuft ein Unterprozess (Profil gilt je Prozess) und speichert seine Signale.
"""
import os
import sys
import json
import pickle
import subprocess
import importlib.util
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
TMP = os.environ.get('TMP', '/tmp')
TF_OF = {'kv6': '2h'}


def child(profile):
    from jaegerbot.strategy import trade_rules as tr
    from jaegerbot.utils.data import load_settings
    from jaegerbot.analysis import optimizer as opt
    assert tr.PROFILE == profile
    spec = importlib.util.spec_from_file_location('p', os.path.join(ROOT, 'forschung', 'pairs', 'evaluate.py'))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    trend = pickle.load(open(P.TREND_EVENTS, 'rb'))
    tf = TF_OF.get(profile, '4h')
    coins = trend['coins']
    opt.tradable_coins = lambda timeframe, settings=None: coins     # gleiches Universum wie das Live-Team
    r = opt.optimize_timeframe(tf, load_settings())
    if 'cfg' not in r:
        pickle.dump({'profile': profile, 'status': r['status']}, open(os.path.join(TMP, f'kv_{profile}.pkl'), 'wb'))
        return
    ev = []
    for e in opt.events_for(r['coins'], tf, r['cfg'], P.START):
        if e['exit_time'] is None:
            continue
        ef = tr.fill_price(e['signal_close'], e['side'], True)
        pnl, _ = tr.trade_pnl(e['side'], 1.0, ef, e['exit_raw'])
        ev.append({'time': e['time'], 'exit_time': e['exit_time'], 'coins': (e['coin'],),
                   'R': pnl / abs(ef - e['sl']), 'risk': 0.95})
    pickle.dump({'profile': profile, 'cfg': r['cfg'], 'is': r['is'], 'events': ev, 'tf': tf, 'n_coins': len(r['coins'])},
                open(os.path.join(TMP, f'kv_{profile}.pkl'), 'wb'))


def main(profiles):
    spec = importlib.util.spec_from_file_location('p', os.path.join(ROOT, 'forschung', 'pairs', 'evaluate.py'))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    from jaegerbot.strategy import trade_rules as tr
    for prof in profiles:
        env = {**os.environ, 'JAEGER_PROFILE': prof, 'PYTHONPATH': os.path.join(ROOT, 'src')}
        subprocess.run([sys.executable, os.path.abspath(__file__), '--child', prof], env=env, cwd=ROOT,
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
    OOS, START = P.OOS, P.START

    def is_best(ev):
        sub = [e for e in ev if e['time'] < OOS]
        bb = None
        for sc in np.round(np.arange(0.05, 3.01, 0.05), 2):
            st = P.simulate(sub, None, scale=float(sc))
            if st['dd'] <= 30 and (bb is None or st['end'] > bb[1]['end']):
                bb = (float(sc), st)
        return bb
    res = {}
    print(f"{'Variante':8s} {'Regel':28s} {'Signale':>7} | IS Kombi mit Long bei DD<=30: {'%/J':>7} {'DD':>5} Faktor")
    for prof in profiles:
        d = pickle.load(open(os.path.join(TMP, f'kv_{prof}.pkl'), 'rb'))
        if 'events' not in d:
            print(f"{prof:8s} {d['status']}")
            continue
        bb = is_best(long_ev + d['events'])
        cg = P.cagr(bb[1], START, OOS)
        res[prof] = {'cfg': d['cfg'], 'factor': bb[0], 'is_cagr': cg, 'is_dd': bb[1]['dd'], 'events': d['events']}
        c = d['cfg']
        print(f"{prof:8s} {c['sides']:2s} Top {100 - c['quantile'] * 100:4.1f} % Trend={int(c['ema_filter'])} {d['tf']:3s}"
              f"       {len(d['events']):7d} |                               {cg:+6.1f}% {bb[1]['dd']:5.1f} {bb[0]}")
    base = res.get('geohtf')
    best = max(res, key=lambda k: res[k]['is_cagr'])
    chosen = best if base is None or res[best]['is_cagr'] > base['is_cagr'] else 'geohtf'
    print(f"\nIS-Auswahl: {chosen}" + ('' if chosen != 'geohtf' else ' (keine Variante schlaegt B0 im IS)'))
    for prof in dict.fromkeys([chosen, 'geohtf']):
        if prof not in res:
            continue
        o = P.simulate([e for e in long_ev + res[prof]['events'] if e['time'] >= OOS], None, scale=res[prof]['factor'])
        res[prof]['oos'] = o
        print(f"OOS 26 W (einmalig, IS-Faktor {res[prof]['factor']}): {prof:8s} {o['pnl']:+.1f} % DD {o['dd']:.1f} n {o['n']}")
    ok = chosen == 'geohtf' or (res[chosen]['oos']['end'] >= res['geohtf']['oos']['end'] and res[chosen]['oos']['dd'] <= 30)
    print('Ergebnis:', f"{chosen} bestaetigt" if ok and chosen != 'geohtf' else
          ('B0 bleibt' if chosen == 'geohtf' else f"{chosen} im OOS nicht bestaetigt -> B0 bleibt"))
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'events'} for k, v in res.items()} | {'chosen': chosen, 'ok': ok},
              open(os.path.join(ROOT, 'forschung', 'kanal_varianten', 'result.json'), 'w'), indent=2, default=float)




def child_unified(profile):
    """Einheits-ANN: Long- und Short-Regel getrennt aus EINEM Modell (Profil profile) im IS waehlen."""
    import itertools
    from jaegerbot.strategy import trade_rules as tr
    from jaegerbot.utils.data import load_settings
    from jaegerbot.analysis import optimizer as opt
    spec = importlib.util.spec_from_file_location('p', os.path.join(ROOT, 'forschung', 'pairs', 'evaluate.py'))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    coins = pickle.load(open(P.TREND_EVENTS, 'rb'))['coins']
    s = load_settings()
    out = {}
    for side in ('L', 'S'):
        opt.GRID[:] = [{'sides': side, 'quantile': q, 'ema_filter': e}
                       for q, e in itertools.product(tr.QUANTILE_LEVELS, (False, True))]
        opt.tradable_coins = lambda timeframe, settings=None: coins
        r = opt.optimize_timeframe('4h', s)
        ev = []
        if 'cfg' in r:
            for e in opt.events_for(r['coins'], '4h', r['cfg'], P.START):
                if e['exit_time'] is None:
                    continue
                ef = tr.fill_price(e['signal_close'], e['side'], True)
                pnl, _ = tr.trade_pnl(e['side'], 1.0, ef, e['exit_raw'])
                ev.append({'time': e['time'], 'exit_time': e['exit_time'], 'coins': (e['coin'],),
                           'R': pnl / abs(ef - e['sl']), 'risk': 0.95, 'prio': 0 if side == 'L' else 1})
        out[side] = {'cfg': r.get('cfg'), 'events': ev}
    pickle.dump(out, open(os.path.join(TMP, f'kvE_{profile}.pkl'), 'wb'))


if __name__ == '__main__':
    if sys.argv[1] == '--child':
        child(sys.argv[2])
    elif sys.argv[1] == '--child-unified':
        child_unified(sys.argv[2])
    else:
        main(sys.argv[1:])
