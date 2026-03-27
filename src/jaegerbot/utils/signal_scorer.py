# src/jaegerbot/utils/signal_scorer.py
"""
Signal Quality Scorer — ersetzt die binäre Filter-Kaskade durch ein
gewichtetes Punktesystem (0–10 Punkte).

Kernidee: Kein einzelner Indikator kann einen Trade mehr alleine BLOCKIEREN.
Stattdessen tragen alle Komponenten zum Gesamt-Score bei.
Ein starkes ANN-Signal kann einen gegen-trend SuperTrend ausgleichen,
starker ADX kann schwaches Volumen kompensieren usw.

Inspiration aus anderen Bots:
  - dnabot  → Konfidenz-Scores statt harte Schwellen
  - dbot    → Regime-Awareness (ADX-Gewichtung)
  - apexbot → Multi-Signal-Scoring (RADAR-Konzept)

Standard-Gewichtung (Summe = 10):
  ANN-Konfidenz   : 4 Punkte  (stärkste Komponente — das Herzstück)
  SuperTrend      : 2 Punkte  (Trend-Bestätigung, kein Hard-Block mehr)
  ADX             : 2 Punkte  (Trendstärke / Marktregime)
  Volumen         : 1 Punkt   (Liquiditätsqualität)
  Volatilität     : 1 Punkt   (Marktruhe / keine Spike-Trades)
"""

from dataclasses import dataclass
from typing import Optional


DEFAULT_WEIGHTS = {
    'ann_weight': 4.0,
    'st_weight': 2.0,
    'adx_weight': 2.0,
    'volume_weight': 1.0,
    'volatility_weight': 1.0,
}

DEFAULT_MIN_SIGNAL_SCORE = 5.5


@dataclass
class ScoreBreakdown:
    ann: float = 0.0
    supertrend: float = 0.0
    adx: float = 0.0
    volume: float = 0.0
    volatility: float = 0.0

    @property
    def total(self) -> float:
        return self.ann + self.supertrend + self.adx + self.volume + self.volatility

    def to_dict(self) -> dict:
        return {
            'ann': round(self.ann, 2),
            'supertrend': round(self.supertrend, 2),
            'adx': round(self.adx, 2),
            'volume': round(self.volume, 2),
            'volatility': round(self.volatility, 2),
            'total': round(self.total, 2),
        }

    def __str__(self) -> str:
        d = self.to_dict()
        return (
            f"Score {d['total']:.1f}/10 "
            f"[ANN={d['ann']:.1f} "
            f"ST={d['supertrend']:.1f} "
            f"ADX={d['adx']:.1f} "
            f"Vol={d['volume']:.1f} "
            f"Vola={d['volatility']:.1f}]"
        )


class SignalScorer:
    """
    Bewertet die Signalqualität auf einer Skala von 0–10 Punkten.

    Ein Trade wird erlaubt, wenn score.total >= min_signal_score.
    Kein Filter kann einen Trade mehr alleine hard-blocken.
    Die Gewichte können über die Config (scoring-Sektion) angepasst werden.
    """

    def __init__(self, weights: Optional[dict] = None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def score(
        self,
        prediction: float,
        pred_threshold: float,
        side: str,                  # 'buy' oder 'sell'
        st_direction: float,        # 1.0 = bullisch, -1.0 = bärisch
        adx: float,
        volume_ratio: float,        # current_volume / rolling_avg_volume (20 Kerzen)
        atr_normalized: float,      # aktueller normalisierter ATR
        avg_atr_normalized: float,  # 50-Kerzen Rolling-Mittel des norm. ATR
    ) -> ScoreBreakdown:
        b = ScoreBreakdown()

        # ── 1. ANN-Konfidenz (0 … ann_weight) ──────────────────────────────
        # Wie weit ist die Vorhersage ÜBER der Schwelle? → Normiert auf 0–1
        # Knappes Signal (0.601 bei threshold=0.6) → fast 0 Punkte
        # Starkes Signal (0.85 bei threshold=0.6)  → fast 4 Punkte
        if side == 'buy':
            raw = (prediction - pred_threshold) / max(1.0 - pred_threshold, 1e-9)
        else:  # sell
            raw = ((1.0 - pred_threshold) - prediction) / max(1.0 - pred_threshold, 1e-9)
        confidence = max(0.0, min(1.0, raw))
        b.ann = confidence * self.weights['ann_weight']

        # ── 2. SuperTrend-Ausrichtung (0 oder st_weight) ───────────────────
        # Binär: Ausgerichtet → volle Punkte, Gegentrend → 0 Punkte.
        # KEIN HARD-BLOCK mehr — der Score sinkt nur, wird aber nicht null.
        if (side == 'buy' and st_direction == 1.0) or \
           (side == 'sell' and st_direction == -1.0):
            b.supertrend = self.weights['st_weight']

        # ── 3. ADX-Trendstärke (0 … adx_weight, graduiert) ─────────────────
        # Schwacher Markt (ADX < 15) → 0, starker Trend (ADX >= 35) → voll
        w = self.weights['adx_weight']
        if adx >= 35:
            b.adx = w
        elif adx >= 25:
            b.adx = w * 0.75
        elif adx >= 20:
            b.adx = w * 0.50
        elif adx >= 15:
            b.adx = w * 0.25
        # else: b.adx bleibt 0

        # ── 4. Volumen-Qualität (0 … volume_weight, graduiert) ─────────────
        v = self.weights['volume_weight']
        if volume_ratio >= 1.2:
            b.volume = v             # Überdurchschnittlich → voll
        elif volume_ratio >= 1.0:
            b.volume = v * 0.75      # Leicht überdurchschnittlich
        elif volume_ratio >= 0.8:
            b.volume = v * 0.50      # Schwaches aber akzeptables Volumen
        # else: unter 80% → 0 Punkte

        # ── 5. Volatilitäts-Qualität (0 … volatility_weight, invertiert) ───
        # Weniger Volatilität = mehr Punkte (saubereres Marktumfeld)
        # Extreme Spikes (ATR > 2× Durchschnitt) → 0 Punkte
        vv = self.weights['volatility_weight']
        if avg_atr_normalized > 0:
            ratio = atr_normalized / avg_atr_normalized
            if ratio <= 1.0:
                b.volatility = vv        # Normal oder ruhig
            elif ratio <= 1.5:
                b.volatility = vv * 0.75
            elif ratio <= 2.0:
                b.volatility = vv * 0.50
            # else: extremer Spike → 0
        else:
            b.volatility = vv * 0.5  # Neutral wenn noch keine Referenz

        return b
