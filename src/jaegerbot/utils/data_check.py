# src/jaegerbot/utils/data_check.py
"""Datenpruefung fuer Kerzen -- dieselben Regeln fuer Download, Training, Backtest und Live.

Lieber nicht handeln/trainieren als auf Daten mit Loechern oder veraltetem Ende: ein fehlender
Abschnitt verschiebt Indikatoren, Labels und SL/TP-Pfade unbemerkt.
"""
import pandas as pd

# Mindesthistorie fuer eine Config: 26 W IS-Minimum + 26 W OOS + Einschwingen
MIN_DAYS_FOR_CONFIG = 7 * 52 + 30
MAX_GAP_CANDLES = 0          # Bitget-Historie ist fuer 2h/4h/6h lueckenlos (geprueft 25.10.2026)


def gaps(df, timeframe):
    """Liste (Zeitpunkt, fehlende Kerzen) aller Luecken im Index."""
    if len(df) < 2:
        return []
    tf = pd.Timedelta(timeframe)
    d = df.index.to_series().diff().dropna()
    bad = d[d != tf]
    return [(ts, int(round(delta / tf)) - 1) for ts, delta in bad.items()]


def check_candles(df, timeframe, min_candles=0, max_age=None, now=None):
    """Liefert Liste von Problemen (leer = in Ordnung).

    max_age: wie alt die letzte GESCHLOSSENE Kerze hoechstens sein darf (Kerzenschluss bis jetzt).
    """
    problems = []
    if df is None or df.empty:
        return ['keine Kerzen']
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        problems.append('Index unsortiert oder doppelt')
    g = gaps(df, timeframe)
    missing = sum(n for _, n in g if n > 0)
    if missing > MAX_GAP_CANDLES:
        worst = max(g, key=lambda x: x[1])
        problems.append(f"{missing} fehlende Kerzen in {len(g)} Luecke(n), groesste {worst[1]} vor {worst[0]}")
    if len(df) < min_candles:
        problems.append(f"nur {len(df)} Kerzen (mind. {min_candles})")
    if df[['open', 'high', 'low', 'close']].isna().any().any() or (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        problems.append('fehlende oder nicht positive Preise')
    if max_age is not None:
        now = now or pd.Timestamp.now(tz='UTC')
        age = now - (df.index[-1] + pd.Timedelta(timeframe))
        if age > max_age:
            problems.append(f"letzte Kerze veraltet ({df.index[-1]}, {age} alt)")
    return problems


def enough_history_for_config(df, timeframe):
    if df is None or df.empty:
        return False
    return (df.index[-1] - df.index[0]).days >= MIN_DAYS_FOR_CONFIG
