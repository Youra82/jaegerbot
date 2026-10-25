#!/usr/bin/env python3
"""Taeglicher Selbstcheck des jaegerbot (vom master_runner gestartet, 1x/Tag ab SEND_HOUR UTC).

Prueft und meldet per Telegram:
1. Daten: Kerzen der aktiven Strategien lueckenlos und aktuell
2. Modell: Ensemble fuer das laufende Quartal vorhanden
3. Live vs. Backtest (letzte 7 Tage): Einstiege gleich / fehlt live / nur live, Ausstiege
4. Schutz: offene Positionen ohne Stop-Loss, Positionen ohne zugehoerige Strategie
5. Mini-Paritaetstest: Live-Entscheidung == Backtest-Signal auf den letzten Kerzen

Aufruf: python daily_self_check.py [--force] [--send] [--days 7]
"""
import os
import sys
import json
import time
import argparse
import pickle
import subprocess

import ccxt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.data import update_cache, load_settings, symbol_of
from jaegerbot.utils.data_check import check_candles
from jaegerbot.utils.telegram import send_message
from jaegerbot.utils.trade_manager import load_state, strategy_key, decide_entry, FETCH_CANDLES
from jaegerbot.strategy import model as mdl
from jaegerbot.strategy.run import load_config
from jaegerbot.analysis.walkforward import refresh_predictions
from jaegerbot.analysis.backtester import run_portfolio, signal_events, strategy_frame
from jaegerbot.utils.data import load_data

SEND_HOUR = 8
MARKER = os.path.join(PROJECT_ROOT, 'artifacts', 'db', '.daily_self_check')
PRODUCT = 'USDT-FUTURES'


def _secrets():
    """Schluessel aus Umgebungsvariablen (Tests vom Desktop) oder secret.json."""
    if os.environ.get('JAEGER_API_KEY'):
        return ({'apiKey': os.environ['JAEGER_API_KEY'], 'secret': os.environ.get('JAEGER_API_SECRET'),
                 'password': os.environ.get('JAEGER_API_PASSWORD')}, {})
    with open(os.path.join(PROJECT_ROOT, 'secret.json'), 'r', encoding='utf-8') as f:
        s = json.load(f)
    return s['jaegerbot'][0], s.get('telegram', {})


def _paginate(call, base, list_key):
    rows, cursor = [], None
    for _ in range(50):
        data = call(dict(base, **({'idLessThan': cursor} if cursor else {}))).get('data') or {}
        batch = data.get(list_key) or []
        rows.extend(batch)
        end_id = data.get('endId')
        if not batch or not end_id or end_id == cursor or len(batch) < 100:
            break
        cursor = end_id
        time.sleep(0.2)
    return rows


def fetch_live(acc, since):
    ex = ccxt.bitget({**{k: acc.get(k) for k in ('apiKey', 'secret', 'password')},
                      'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    closed = _paginate(ex.privateMixGetV2MixPositionHistoryPosition,
                       {'productType': PRODUCT, 'startTime': str(int((since - pd.Timedelta(days=30)).timestamp() * 1000)),
                        'limit': '100'}, 'list')
    open_pos = ex.privateMixGetV2MixPositionAllPosition({'productType': PRODUCT, 'marginCoin': 'USDT'})['data'] or []
    plans = []
    for pt in ('profit_loss', 'normal_plan'):
        d = ex.privateMixGetV2MixOrderOrdersPlanPending({'productType': PRODUCT, 'planType': pt})['data'] or {}
        plans += [dict(o, planType=pt) for o in (d.get('entrustedList') or [])]
    acct = ex.privateMixGetV2MixAccountAccounts({'productType': PRODUCT})['data']
    equity = next((float(a['accountEquity']) for a in acct if a['marginCoin'] == 'USDT'), 0.0)
    ms = lambda x: pd.Timestamp(int(x), unit='ms', tz='UTC')
    trades = [dict(symbol=p['symbol'], side=p['holdSide'], entry_time=ms(p['ctime']), exit_time=ms(p['utime']),
                   entry=float(p['openAvgPrice']), exit=float(p['closeAvgPrice']), pnl=float(p['netProfit']))
              for p in closed if ms(p['ctime']) >= since]
    trades += [dict(symbol=p['symbol'], side=p['holdSide'], entry_time=ms(p['cTime']), exit_time=None,
                    entry=float(p['openPriceAvg']), exit=None, pnl=float(p['unrealizedPL']))
               for p in open_pos if ms(p['cTime']) >= since]
    return trades, open_pos, plans, equity


def positions_without_sl(open_pos, plans):
    """Wie fibot/stbot: SL = Schliess-Trigger auf der Verlustseite oder stopLossTriggerPrice."""
    flags = []
    for p in open_pos:
        side, sym = p.get('holdSide'), p.get('symbol')
        mark = float(p.get('markPrice') or p.get('openPriceAvg') or 0)
        has_sl = False
        for o in plans:
            if o.get('symbol') != sym or o.get('tradeSide') == 'open':
                continue
            if o.get('posSide') and side and o['posSide'] != side:
                continue
            if str(o.get('stopLossTriggerPrice') or '0') not in ('', '0', '0.0'):
                has_sl = True
                break
            trig = float(o.get('triggerPrice') or 0)
            if trig > 0 and mark > 0 and ((side == 'long' and trig < mark) or (side == 'short' and trig > mark)):
                has_sl = True
                break
        if not has_sl:
            flags.append(f"❗ {sym} {side} OHNE Stop-Loss (Kurs {mark:g})")
    return flags


def mini_parity(coin, tf, cfg, n=30):
    raw = load_data(symbol_of(coin), tf)
    sf = strategy_frame(coin, tf)
    if sf is None or raw.empty:
        return ['keine Daten']
    idx = sf.index[sf['pL'].notna()][-n:]
    bt = {e['time'] - pd.Timedelta(tf): e['side'] for e in signal_events(coin, tf, cfg, start=idx[0] + pd.Timedelta(tf))}
    out = []
    for ts, i in zip(idx, raw.index.get_indexer(idx)):
        d = decide_entry(raw.iloc[max(0, i + 1 - FETCH_CANDLES):i + 1], tf, cfg)
        if d.get('error') or d['side'] != bt.get(ts):
            out.append(f"{ts:%d.%m %H:%M} live {d.get('error') or d['side']} / BT {bt.get(ts)}")
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')   # Emojis auch in Windows-Konsolen/Logdateien
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true')
    p.add_argument('--send', action='store_true', help='auch bei --force per Telegram senden')
    p.add_argument('--days', type=int, default=7)
    a = p.parse_args()
    now = pd.Timestamp.now(tz='UTC')
    today = f"{now:%Y-%m-%d}"
    if not a.force:
        if now.hour < SEND_HOUR:
            return
        if os.path.exists(MARKER) and open(MARKER).read().strip() == today:
            return
    s = load_settings()
    lt = s['live_trading_settings']
    team = [x for x in lt.get('active_strategies', []) if x.get('active', True)]
    lines, warn = [f"🩺 jaegerbot Selbstcheck {now:%d.%m.%Y %H:%M} UTC"], []

    # 1+2 Daten, Modell, Backtest-Signale und Mini-Paritaet je Profil (Profil gilt je Prozess -> Unterprozess)
    since = now - pd.Timedelta(days=a.days)
    profiles = sorted({x.get('profile', 'trend') for x in team}, key=lambda p: (p != 'trend', p))
    events, n_strat, n_bad = [], 0, 0
    for prio, prof in enumerate(profiles):
        out = os.path.join(PROJECT_ROOT, 'artifacts', 'db', f'.selfcheck_{prof}.pkl')
        env = {**os.environ, 'JAEGER_PROFILE': prof, 'PYTHONPATH': os.path.join(PROJECT_ROOT, 'src')}
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), '--dump', prof, out, since.isoformat(), str(prio)],
                             cwd=PROJECT_ROOT, env=env)
        if rc != 0 or not os.path.exists(out):
            warn.append(f"❗ Selbstcheck-Teil {prof} fehlgeschlagen (rc={rc})")
            continue
        with open(out, 'rb') as f:
            d = pickle.load(f)
        warn += d['warn']
        events += d['events']
        n_strat += d['n']
        n_bad += d['parity_bad']

    # 3 Live vs Backtest
    acc, tg = _secrets()
    live, open_pos, plans, equity = fetch_live(acc, since)
    sim = run_portfolio([], max(equity, 1.0), lt['risk_per_trade_pct'], lt['max_open_positions'],
                        lt['max_leverage'], events=events)
    st = sim['trades']
    sim_rows = [dict(symbol=r['coin'] + 'USDT', side=r['side'], time=r['time'], exit_time=r['exit_time'], tf=r['timeframe'])
                for _, r in st.iterrows()] if len(st) else []
    sim_rows += [dict(symbol=o['coin'] + 'USDT', side=o['side'], time=o['time'], exit_time=None, tf=o['timeframe'])
                 for o in sim['open_positions']]
    used, matched, missing = set(), 0, []
    for r in sim_rows:
        hit = next((k for k, t in enumerate(live) if k not in used and t['symbol'] == r['symbol'] and t['side'] == r['side']
                    and pd.Timedelta(0) <= t['entry_time'] - r['time'] <= pd.Timedelta(minutes=20)), None)
        if hit is None:
            missing.append(f"{r['symbol']} {r['side']} {r['time']:%d.%m %H:%M}")
            continue
        used.add(hit)
        matched += 1
        t = live[hit]
        # SL/TP loesen live innerhalb der Kerze aus, der Backtest bucht zum Kerzenschluss
        tol = pd.Timedelta(r['tf']) + pd.Timedelta(minutes=20)
        if (t['exit_time'] is None) != (r['exit_time'] is None) or (
                t['exit_time'] is not None and abs(t['exit_time'] - r['exit_time']) > tol):
            warn.append(f"⚠ Ausstieg {r['symbol']} {r['side']}: live {t['exit_time']} / BT {r['exit_time']}")
    extra = [f"{t['symbol']} {t['side']} {t['entry_time']:%d.%m %H:%M}" for k, t in enumerate(live) if k not in used]
    lines.append(f"Konto {equity:.2f} USDT | Team {n_strat} ({', '.join(profiles)}) | {a.days} Tage: {matched} gleich, "
                 f"{len(missing)} fehlen live, {len(extra)} nur live")
    if missing:
        warn.append("⚠ Fehlt live: " + ', '.join(missing[:8]))
    if extra:
        warn.append("⚠ Nur live: " + ', '.join(extra[:8]))
    live_pnl = sum(t['pnl'] for t in live if t['exit_time'] is not None)
    sim_pnl = float(st['pnl'].sum()) if len(st) else 0.0
    lines.append(f"Realisiert live {live_pnl:+.2f} USDT | Backtest {sim_pnl:+.2f} USDT")

    # 4 Schutz und Waisen
    warn += positions_without_sl(open_pos, plans)
    claimed = set()
    for x in team:
        prof = x.get('profile', 'trend')
        sdir = os.path.join(PROJECT_ROOT, 'artifacts', 'db' + ('' if prof == 'trend' else f'_{prof}'))
        try:
            with open(os.path.join(sdir, f"state_{strategy_key(x['symbol'], x['timeframe'])}.json"), encoding='utf-8') as f:
                if json.load(f).get('position'):
                    claimed.add(x['symbol'].replace('/USDT:USDT', 'USDT'))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    for p_ in open_pos:
        if p_['symbol'] not in claimed:
            warn.append(f"❗ Position {p_['symbol']} {p_['holdSide']} gehoert keiner Strategie (Waise)")

    # 5 Mini-Paritaet (in den Profil-Unterprozessen gerechnet)
    lines.append(f"Paritaet Live-Code = Backtest: {n_strat - n_bad}/{n_strat}")

    lines += warn if warn else ["✅ Keine Auffaelligkeiten."]
    text = '\n'.join(lines)
    print(text)
    if not a.force or a.send:
        try:
            send_message(tg.get('bot_token'), tg.get('chat_id'), text)
        except Exception as e:
            print(f"Telegram fehlgeschlagen: {e}")
    if not a.force:
        os.makedirs(os.path.dirname(MARKER), exist_ok=True)
        with open(MARKER, 'w') as f:
            f.write(today)


def dump_profile(profile, out, since, prio):
    """Unterprozess (JAEGER_PROFILE gesetzt): Teil des Selbstchecks fuer die Strategien eines Profils."""
    from jaegerbot.strategy import trade_rules as tr
    assert tr.PROFILE == profile, f"Profil {tr.PROFILE} != {profile}"
    now = pd.Timestamp.now(tz='UTC')
    team = [x for x in load_settings()['live_trading_settings'].get('active_strategies', [])
            if x.get('active', True) and x.get('profile', 'trend') == profile]
    warn, events, bad = [], [], 0
    tag = '' if profile == 'trend' else f' [{profile}]'
    for tf in sorted({x['timeframe'] for x in team}):
        q = mdl.quarter_starts(now, now)[0]
        if q not in mdl.available_cutoffs(tf) and now > q + pd.Timedelta(days=2):
            warn.append(f"❗ Kein Modell fuer {tf}{tag} im laufenden Quartal (ab {q:%d.%m.}) -- Quartals-Pipeline pruefen")
    for x in team:
        coin, tf = x['symbol'].split('/')[0], x['timeframe']
        df = update_cache(x['symbol'], tf)
        problems = check_candles(df, tf, max_age=pd.Timedelta(hours=1) + pd.Timedelta(tf))
        if problems:
            warn.append(f"❗ Daten {coin} {tf}: {'; '.join(problems)}")
        refresh_predictions(coin, tf)
        try:
            cfg = load_config(x['symbol'], tf)['strategy']
        except FileNotFoundError:
            warn.append(f"❗ Config fehlt{tag}: {coin} {tf}")
            continue
        for e in signal_events(coin, tf, cfg, start=pd.Timestamp(since)):
            e['key'] = f"{coin}_{tf}" + ('' if profile == 'trend' else f'_{profile}')
            e['prio'] = int(prio)
            events.append(e)
        diff = mini_parity(coin, tf, cfg, n=12)
        if diff:
            bad += 1
            warn.append(f"❗ Paritaet{tag} {coin} {tf}: {diff[0]}")
    with open(out, 'wb') as f:
        pickle.dump({'warn': warn, 'events': events, 'n': len(team), 'parity_bad': bad}, f)


if __name__ == '__main__':
    if len(sys.argv) >= 6 and sys.argv[1] == '--dump':
        dump_profile(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()
