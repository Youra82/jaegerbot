# src/jaegerbot/utils/data.py
"""Kerzen-Cache fuer Training, Backtest und Portfolio-Simulation (eine Quelle fuer alle)."""
import os
import json
import logging
import pandas as pd

from jaegerbot.utils.exchange import Exchange

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache')
logger = logging.getLogger(__name__)

_exchange = None


def _get_exchange():
    """Oeffentliche Kerzen brauchen keinen Key; Exchange nur einmal pro Prozess laden."""
    global _exchange
    if _exchange is None:
        _exchange = Exchange({})
    return _exchange


def cache_file(symbol, timeframe):
    return os.path.join(CACHE_DIR, f"{symbol.replace('/', '-').replace(':', '-')}_{timeframe}.csv")


def symbol_of(coin):
    return f"{coin}/USDT:USDT"


def drop_running_candle(df, timeframe):
    if df.empty:
        return df
    tf = pd.Timedelta(timeframe)
    if df.index[-1] + tf > pd.Timestamp.now(tz='UTC'):
        return df.iloc[:-1]
    return df


def update_cache(symbol, timeframe, start='2021-01-01'):
    """Laedt fehlende Kerzen nach (ab letzter Cache-Kerze) und speichert nur geschlossene Kerzen."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_file(symbol, timeframe)
    old = pd.DataFrame()
    if os.path.exists(path):
        old = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        old.index = pd.to_datetime(old.index, utc=True)
    if symbol not in (_get_exchange().markets or {}):
        logger.warning(f"{symbol}: nicht (mehr) bei Bitget gelistet.")
        return old
    fetch_from = (old.index[-1] - pd.Timedelta(days=1)).strftime('%Y-%m-%d') if not old.empty else start
    end = (pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    new = _get_exchange().fetch_historical_ohlcv(symbol, timeframe, fetch_from, end, quiet=True)
    if new.empty and old.empty:
        return old
    df = pd.concat([old, new]) if not old.empty else new
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = drop_running_candle(df, timeframe)
    df.to_csv(path)
    return df


def load_data(symbol, timeframe, start=None, end=None):
    """Nur aus dem Cache (update_cache vorher); start/end inklusive, als Datum oder Timestamp."""
    path = cache_file(symbol, timeframe)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz='UTC') if pd.Timestamp(start).tzinfo is None else df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz='UTC') if pd.Timestamp(end).tzinfo is None else df.index <= pd.Timestamp(end)]
    return df


LISTING_FILE = os.path.join(PROJECT_ROOT, 'data', 'listing_check.json')


def established_coins(coins, min_days):
    """Coins, fuer die Bitget vor min_days Tagen schon eine Tageskerze hatte (Listing-Alter).

    Bitget liefert kein Listing-Datum in den Marktdaten (launchTime leer) -- daher Abfrage der
    Tageskerze am Stichtag. Ergebnis je Tag zwischengespeichert. Im Test 25.10.2026 hatte das
    ANN nur auf etablierten Coins eine Kante; auf jungen Altcoins ~0 R.
    """
    today = f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}"
    cache = {}
    if os.path.exists(LISTING_FILE):
        with open(LISTING_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    if cache.get('date') != today or cache.get('min_days') != min_days:
        cache = {'date': today, 'min_days': min_days, 'coins': {}}
    ex = _get_exchange().exchange
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=min_days)
    changed = False
    for coin in coins:
        if coin in cache['coins']:
            continue
        sym = symbol_of(coin)
        ok = False
        if sym in (_get_exchange().markets or {}):
            try:
                d = ex.fetch_ohlcv(sym, '1d', since=int(cutoff.timestamp() * 1000), limit=5)
                ok = bool(d) and d[0][0] <= int((cutoff + pd.Timedelta(days=30)).timestamp() * 1000)
            except Exception as e:
                logger.warning(f"Listing-Pruefung {coin}: {e}")
                continue    # unbekannt -> beim naechsten Mal erneut pruefen, nicht als 'jung' speichern
        cache['coins'][coin] = ok
        changed = True
    if changed:
        os.makedirs(os.path.dirname(LISTING_FILE), exist_ok=True)
        with open(LISTING_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=1)
    return [c for c in coins if cache['coins'].get(c)]


def load_settings():
    with open(os.path.join(PROJECT_ROOT, 'settings.json'), 'r', encoding='utf-8') as f:
        return json.load(f)
