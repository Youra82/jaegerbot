# src/jaegerbot/utils/supertrend_indicator.py (FINALE KORREKTUR V6.1: Import 'ta')
import pandas as pd
import numpy as np
import ta # NEU: Das fehlende Modul importieren

# Implementierung der SuperTrend Logik aus ta-lib, da der Import fehlschlägt.

class SuperTrendLocal:
    """
    SuperTrend Indicator implementation.
    Reference: https://www.tradingview.com/script/hlfVjS8F-SuperTrend/
    """
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10, multiplier: float = 3.0, fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._multiplier = multiplier
        self._fillna = fillna
        self._run()

    def _run(self):
        # KORRIGIERT: ta.volatility.average_true_range wird jetzt korrekt aufgerufen
        # Muss als Series erstellt werden, um den Index zu behalten
        atr_indicator = pd.Series(ta.volatility.average_true_range(self._high, self._low, self._close, window=self._window, fillna=self._fillna))

        # Calculate HL2 (Midpoint)
        hl2 = (self._high + self._low) / 2

        # Calculate Basic Upper/Lower Bands
        basic_upper_band = hl2 + self._multiplier * atr_indicator
        basic_lower_band = hl2 - self._multiplier * atr_indicator

        # Initialisiere Arrays (die Indizes müssen mit denen der Series übereinstimmen)
        final_upper_band = np.zeros(len(self._close))
        final_lower_band = np.zeros(len(self._close))
        supertrend_series = np.zeros(len(self._close))
        supertrend_direction = np.zeros(len(self._close)) # 1.0 = Up, -1.0 = Down

        # Initialer Wert für die erste Kerze, wo ATR nicht NaN ist
        first_valid_idx = atr_indicator.first_valid_index()
        if first_valid_idx is None:
            # Kann passieren, wenn zu wenige Daten vorhanden sind
            self.supertrend = pd.Series(np.nan, index=self._close.index)
            self.supertrend_direction = pd.Series(np.nan, index=self._close.index)
            return

        first_i = self._close.index.get_loc(first_valid_idx)
        final_upper_band[first_i] = basic_upper_band.iloc[first_i]
        final_lower_band[first_i] = basic_lower_band.iloc[first_i]
        # Initialisiere Supertrend für den ersten validen Punkt
        supertrend_series[first_i] = final_upper_band[first_i]
        supertrend_direction[first_i] = -1.0 # Standardmäßig short

        for i in range(first_i + 1, len(self._close)):
            # Update Final Upper Band
            if basic_upper_band.iloc[i] < final_upper_band[i-1] or self._close.iloc[i-1] > final_upper_band[i-1]:
                final_upper_band[i] = basic_upper_band.iloc[i]
            else:
                final_upper_band[i] = final_upper_band[i-1]

            # Update Final Lower Band
            if basic_lower_band.iloc[i] > final_lower_band[i-1] or self._close.iloc[i-1] < final_lower_band[i-1]:
                final_lower_band[i] = basic_lower_band.iloc[i]
            else:
                final_lower_band[i] = final_lower_band[i-1]

            # Determine Supertrend Value and Direction
            if supertrend_series[i-1] == final_upper_band[i-1]:
                if self._close.iloc[i] <= final_upper_band[i]:
                    supertrend_series[i] = final_upper_band[i]
                    supertrend_direction[i] = -1.0 # Down
                else:
                    supertrend_series[i] = final_lower_band[i]
                    supertrend_direction[i] = 1.0 # Up
            elif supertrend_series[i-1] == final_lower_band[i-1]:
                if self._close.iloc[i] >= final_lower_band[i]:
                    supertrend_series[i] = final_lower_band[i]
                    supertrend_direction[i] = 1.0 # Up
                else:
                    supertrend_series[i] = final_upper_band[i]
                    supertrend_direction[i] = -1.0 # Down
            
            # Sollte aufgrund der Initialisierung selten passieren, ist aber eine Sicherung
            elif self._close.iloc[i] > supertrend_series[i-1]:
                 supertrend_series[i] = final_lower_band[i]
                 supertrend_direction[i] = 1.0
            else:
                 supertrend_series[i] = final_upper_band[i]
                 supertrend_direction[i] = -1.0

        self.supertrend = pd.Series(supertrend_series, index=self._close.index)
        self.supertrend_direction = pd.Series(supertrend_direction, index=self._close.index)

    def get_supertrend_direction(self) -> pd.Series:
        return self.supertrend_direction
