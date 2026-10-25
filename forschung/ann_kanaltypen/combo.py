"""Hypothese C: Live-Trend-Long + geohtf-Short-Regel (JAEGER_PROFILE=geohtf)."""
import os, sys, json, pickle, importlib.util
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src')); os.chdir(ROOT)
from jaegerbot.strategy import trade_rules as tr
from jaegerbot.analysis import optimizer as opt
spec = importlib.util.spec_from_file_location('ev', os.path.join(ROOT, 'forschung', 'ann_kanaltypen', 'evaluate.py'))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
P = ev.pairs
assert tr.PROFILE == 'geohtf'
trend = pickle.load(open(P.TREND_EVENTS, 'rb')); coins = trend['coins']
with open(os.path.join(ROOT, 'forschung', 'ann_kanaltypen', 'result_geohtf.json'), encoding='utf-8') as f:
    rule = json.load(f)['rule']
long_ev = ev.to_R(trend['events'], 0.95)
short_ev = ev.to_R(opt.events_for(coins, '4h', rule, P.START), 0.95)
combo = long_ev + short_ev
now = pd.Timestamp.now(tz='UTC')
t_oos = P.simulate(long_ev, None, start=P.OOS)
print(f"Regel Short: {rule} | Long-Signale {len(long_ev)}, Short-Signale {len(short_ev)}")
print(f"Trend allein OOS (0,95 %): {t_oos['pnl']:+.1f} % DD {t_oos['dd']:.1f}")
# K1: Risiko so, dass OOS-DD der Kombination <= OOS-DD Trend allein
k1 = None
for sc in np.round(np.arange(0.05, 2.01, 0.05), 2):
    o = P.simulate(combo, None, start=P.OOS, scale=float(sc))
    if o['dd'] <= t_oos['dd'] and (k1 is None or o['end'] > k1[1]['end']):
        k1 = (float(sc), o)
print(f"C1 OOS bei gleichem DD: Faktor {k1[0]} -> {k1[1]['pnl']:+.1f} % DD {k1[1]['dd']:.1f} n {k1[1]['n']}")
bt, bc = ev.best(long_ev), ev.best(combo)
print(f"C2 DD<=30 seit 2023: Trend {P.cagr(bt[1], P.START, now):+.1f} %/J | Kombi {P.cagr(bc[1], P.START, now):+.1f} %/J (Faktor {bc[0]}, DD {bc[1]['dd']:.1f})")
oc = P.simulate(combo, None, start=P.OOS, scale=bc[0])
print(f"    dabei OOS: Kombi {oc['pnl']:+.1f} % DD {oc['dd']:.1f}")
ts, cs = P.simulate(long_ev, None, start=P.SIDE_A, end=P.SIDE_B), P.simulate(combo, None, start=P.SIDE_A, end=P.SIDE_B)
print(f"C3 Seitwaertsfenster 0,95 %: Trend {ts['pnl']:+.1f} % | Kombi {cs['pnl']:+.1f} % (DD {cs['dd']:.1f})")
yr = {}
for y in range(2023, 2027):
    a, b = pd.Timestamp(f'{y}-01-01', tz='UTC'), pd.Timestamp(f'{y}-12-31', tz='UTC')
    yr[y] = (P.simulate(long_ev, None, start=a, end=b)['pnl'], P.simulate(combo, None, start=a, end=b, scale=bc[0])['pnl'])
print('Jahre (Trend 0,95 % | Kombi Faktor ' + str(bc[0]) + '): ' + ' | '.join(f"{y}: {t:+.0f} / {c:+.0f} %" for y, (t, c) in yr.items()))
res = {'c1': k1[1]['end'] > t_oos['end'], 'c2': bc[1]['end'] > bt[1]['end'], 'c3': cs['end'] > ts['end']}
print('C1-C3:', {k: ('ok' if v else 'NICHT erfuellt') for k, v in res.items()}, '(C4 = Paritaet separat)')
json.dump({'rule': rule, 'c': res, 'oos_same_dd': [k1[0], k1[1]['pnl'], k1[1]['dd']],
           'cagr': [P.cagr(bt[1], P.START, now), P.cagr(bc[1], P.START, now), bc[0]], 'years': yr},
          open(os.path.join(ROOT, 'forschung', 'ann_kanaltypen', 'result_combo.json'), 'w'), indent=2, default=float)
