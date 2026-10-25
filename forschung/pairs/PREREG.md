# Vorregistrierung: Pair-Trading als Seitwärts-Baustein (25.10.2026)

Festgelegt **bevor** gerechnet wird. Keine Parameter-Optimierung, keine Nachjustierung.

## Anlass
User: „Finde einen Weg, auch im Seitwärtsmarkt zu traden.“ Richtungs-Strategien (Trend-ANN, Range-ANN v1/v2) haben
im Seitwärtsmarkt keine Kante. Gesucht ist ein **anderer Mechanismus**, der unabhängig von der Marktrichtung verdient.

## Hypothese (Mechanismus)
Eng verwandte Coins (gemeinsame Treiber: BTC-Beta, Sektor, Liquiditätsströme) bilden kurzfristig eine stationäre
Preisdifferenz. Läuft sie auseinander (Order-Flow-Schock in einem Coin), kehrt sie zurück. Long den relativ billigen,
Short den relativ teuren Coin → marktneutral, verdient an der Differenz, nicht an der Richtung.

## Regeln (alle fest)
- Daten: 4h-Schlusskurse, die 39 Coins des Live-Teams, alle 741 Paare.
- **Formation** (zu jedem Monatsbeginn, nur Vergangenheit): letzte 120 Tage (720 Kerzen) log-Preise.
  β = OLS-Steigung log(A) ~ log(B); Spread s = log A − β·log B; μ, σ = Mittel/Std. von s im Fenster.
  Engle-Granger: Δs_t = a + b·s_{t−1}; Paar zulässig, wenn t(b) < −3,4 und Halbwertszeit −ln2/ln(1+b) ≤ 30 Kerzen.
  Höchstens 20 Paare je Monat (kleinstes t(b)), nur 0,2 ≤ β ≤ 5.
- **Handel** im folgenden Monat mit festen β, μ, σ: z = (s − μ)/σ zum Kerzenschluss.
  Einstieg bei |z| ≥ 2 (z > 2: A short / B long; z < −2: A long / B short), Notional B = β × Notional A.
  Ausstieg bei Rückkehr über 0, Stop bei |z| ≥ 4, Zeitausstieg nach 30 Kerzen. Offene Trades laufen über das Monatsende.
- **Kosten**: 0,11 % je Seite und Coin (Gebühr + Schlupf), also 0,44 % Notional je Paar-Runde (beide Beine).
- **Größe**: 1 R = Verlust beim Stop (2σ Spread-Bewegung); Risiko 1 % des Kapitals je Paar-Trade.
- Portfolio: max. 20 offene Paar-Trades, jeder Coin höchstens in einem offenen Trade.

## Kriterien — alle vier müssen erfüllt sein
1. **OOS** (26.04.2026 – heute), 1 % Risiko: PnL ≥ +10 %, DD ≤ 30 %, ≥ 30 Trades.
2. **Kontrollgruppe**: zufällige Einstiegszeitpunkte in denselben Paaren/Monaten, gleiche Ausstiegsregeln:
   Differenz avgR mit t > 2.
3. **Nutzen**: gemeinsames Konto Trend + Pairs (jeder Coin nur eine Position):
   a) im Seitwärtsfenster 15.03.2024 – 31.07.2026 besser als Trend allein (Trend 0,95 %, Pairs 1 %), und
   b) seit 2023 bei DD ≤ 30 % (Risiko gemeinsam skaliert) höhere Jahresrendite als Trend allein (+34,5 %).
4. **Robustheit** (keine IS-Auswahl, daher ganzer Zeitraum seit 2023): avgR netto > 0 mit t > 2 und in ≥ 60 % der
   Quartale positiv.

Fällt ein Punkt durch: nicht einbauen, Ergebnis dokumentieren.

## Ergebnis (25.10.2026) — NICHT BESTANDEN
- 15.583 Paar-Trades seit 2023 (Ø 19,9 Paare/Monat): netto −0,056 R (t −15,3), Trefferquote 37,6 %; nur 1 von 16 Quartalen positiv.
- **Brutto (ohne Kosten) −0,002 R, t −0,6**: die im Formationsfenster cointegrierten Spreads kehren im Folgemonat
  nicht verlässlich zurück — es gibt keinen Rückkehr-Effekt, den die Kosten auffressen könnten; die Kosten machen
  aus null einen klaren Verlust.
- K1 OOS −45,9 % (DD 52); K2 Kontrolle −0,080 R, ANN-freie Regel nur „weniger schlecht" als Zufall (t 4,6);
  K3 Seitwärtsfenster Pairs −95,6 %; K3b kombiniert −7,5 %/J; K4 nicht erfüllt.
- Simulator-Gegenprobe: Trend allein im R-Simulator +34,5 %/J, DD 29,9 = Bot-Simulator (Rechnung stimmt).
