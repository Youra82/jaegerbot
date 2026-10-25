# src/jaegerbot/strategy/candle_patterns.py
"""Kerzenformen-Bibliothek (forschung/kanal_varianten V7/V8). Jede Form ist 0/1 auf der letzten Kerze i und nutzt nur
die Kerzen i-2..i. Groessen relativ zum ATR bzw. zur Kerzenspanne, Trendkontext aus den 5 Kerzen davor."""
import numpy as np

NAMES = ['hammer', 'inv_hammer', 'shooting_star', 'hanging_man', 'doji', 'dragonfly_doji', 'gravestone_doji',
         'marubozu_bull', 'marubozu_bear', 'spinning_top', 'pin_bull', 'pin_bear', 'engulf_bull', 'engulf_bear',
         'harami_bull', 'harami_bear', 'piercing', 'dark_cloud', 'tweezer_top', 'tweezer_bottom', 'inside_bar',
         'outside_bar', 'kicker_bull', 'kicker_bear', 'morning_star', 'evening_star', 'three_soldiers', 'three_crows']


def candle_patterns(o, h, l, c, atr, prefix):
    n = len(c)
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    up_w = h - np.maximum(o, c)
    lo_w = np.minimum(o, c) - l
    bull, bear = c > o, c < o
    # Trendkontext: Schlusskurs gegenueber dem vor 5 Kerzen
    prior_up = np.zeros(n, bool)
    prior_dn = np.zeros(n, bool)
    prior_up[5:] = c[4:-1] > c[:-5]          # bis Kerze i-1 gestiegen
    prior_dn[5:] = c[4:-1] < c[:-5]
    s1 = lambda x: np.concatenate([[np.nan], x[:-1]])          # Wert der Vorkerze
    s2 = lambda x: np.concatenate([[np.nan, np.nan], x[:-2]])
    o1, h1, l1, c1, b1 = s1(o), s1(h), s1(l), s1(c), s1(body)
    o2, c2, b2 = s2(o), s2(c), s2(body)
    bull1, bear1, bull2, bear2 = c1 > o1, c1 < o1, c2 > o2, c2 < o2
    big = body >= 0.6 * atr
    small = body <= 0.3 * rng
    doji = body <= 0.1 * rng
    hammer_shape = (lo_w >= 2 * body) & (up_w <= 0.3 * body + 0.1 * rng) & ~doji
    inv_shape = (up_w >= 2 * body) & (lo_w <= 0.3 * body + 0.1 * rng) & ~doji
    mid1 = (o1 + c1) / 2
    p = {
        'hammer': hammer_shape & prior_dn,
        'hanging_man': hammer_shape & prior_up,
        'inv_hammer': inv_shape & prior_dn,
        'shooting_star': inv_shape & prior_up,
        'doji': doji,
        'dragonfly_doji': doji & (lo_w >= 0.6 * rng) & (up_w <= 0.1 * rng),
        'gravestone_doji': doji & (up_w >= 0.6 * rng) & (lo_w <= 0.1 * rng),
        'marubozu_bull': bull & big & (up_w + lo_w <= 0.1 * rng),
        'marubozu_bear': bear & big & (up_w + lo_w <= 0.1 * rng),
        'spinning_top': small & ~doji & (up_w >= body) & (lo_w >= body),
        'pin_bull': (lo_w >= 0.66 * rng) & (rng >= atr),
        'pin_bear': (up_w >= 0.66 * rng) & (rng >= atr),
        'engulf_bull': bull & bear1 & (c >= o1) & (o <= c1) & (body > b1),
        'engulf_bear': bear & bull1 & (c <= o1) & (o >= c1) & (body > b1),
        'harami_bull': bull & bear1 & (np.maximum(o, c) <= o1) & (np.minimum(o, c) >= c1) & (b1 >= 0.6 * atr),
        'harami_bear': bear & bull1 & (np.maximum(o, c) <= c1) & (np.minimum(o, c) >= o1) & (b1 >= 0.6 * atr),
        'piercing': bull & bear1 & (o < c1) & (c > mid1) & (c < o1),
        'dark_cloud': bear & bull1 & (o > c1) & (c < mid1) & (c > o1),
        'tweezer_top': prior_up & (np.abs(h - h1) <= 0.1 * atr) & bull1 & bear,
        'tweezer_bottom': prior_dn & (np.abs(l - l1) <= 0.1 * atr) & bear1 & bull,
        'inside_bar': (h < h1) & (l > l1),
        'outside_bar': (h > h1) & (l < l1),
        'kicker_bull': bear1 & bull & (o > o1) & big,
        'kicker_bear': bull1 & bear & (o < o1) & big,
        'morning_star': bear2 & (b2 >= 0.6 * atr) & (s1(body) <= 0.3 * s1(rng)) & bull & (c > (o2 + c2) / 2),
        'evening_star': bull2 & (b2 >= 0.6 * atr) & (s1(body) <= 0.3 * s1(rng)) & bear & (c < (o2 + c2) / 2),
        'three_soldiers': bull & bull1 & bull2 & (c > c1) & (c1 > c2) & (body >= 0.5 * atr) & (b1 >= 0.5 * atr),
        'three_crows': bear & bear1 & bear2 & (c < c1) & (c1 < c2) & (body >= 0.5 * atr) & (b1 >= 0.5 * atr),
    }
    return {f"{prefix}{k}": np.nan_to_num(np.asarray(v, float)) for k, v in p.items()}
