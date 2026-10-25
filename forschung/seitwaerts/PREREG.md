# Vorregistrierung: Seitwärts-Profil für den jaegerbot (25.10.2026)

Festgelegt **bevor** ein Ergebnis vorliegt. Keine Nachjustierung nach Sicht der Ergebnisse.

## Anlass
Die 4h-Trend-Regel (Long, Top 1 %, EMA20 > EMA50, TP 4 ATR / SL 2 ATR) verdient fast nur in Trendphasen.
Equity März 2024 – Juli 2026 seitwärts bis leicht fallend (220 → 197). User-Ziel: auch im Seitwärtsmarkt profitabel.

## Hypothese (Mechanismus)
In Phasen ohne Trend pendelt der Kurs innerhalb einer Spanne. Ein ANN, das auf ein **symmetrisches, kurzes Ziel**
trainiert wird, kann Kerzen erkennen, nach denen der Kurs eher +1,5 ATR als −1,5 ATR erreicht (Long) bzw. umgekehrt
(Short) — z. B. an Spannenrändern nach Überdehnung. Feste Mean-Reversion-Regeln (Grid, Squeeze-Fade, ltbbot-Envelope)
hatten keine Kante; neu ist die gelernte Auswahl.

## Aufbau (identisch zum Trend-Profil bis auf das Ziel)
- Profil `range`: SL = 1,5 × ATR, TP = 1,5 × ATR (RR 1), Zeitausstieg nach 8 Kerzen (4h).
- Gleiche 46 Features, gleiches Ensemble (3 MLPs je Seite), gleiche Quartals-Cutoffs ab 2023-01-01, gleiche Kosten
  (0,06 % Gebühr + 0,05 % Schlupf je Seite).
- Timeframe 4h, Universum = die 39 etablierten Coins des Live-Teams.
- Regel-Raster wie im Optimizer: Seiten L/S/LS × Schwelle Top 10/5/2,5/1 % × Trendfilter an/aus; IS-Wahl nach avgR
  (≥ 100 IS-Trades, IS-PnL > 0). IS = 2023-01-01 bis 26.04.2026, OOS = 26.04.2026 bis heute.

## Kriterien — alle vier müssen erfüllt sein
1. **OOS-Bestätigung** der IS-gewählten Regel: OOS ≥ +10 %, OOS-DD ≤ 30 %, ≥ 30 OOS-Trades (1 % Risiko, max. 20 Pos.).
2. **Kontrollgruppe**: ANN-Signale der gewählten Regel vs. zufällige Einstiege (gleiche Seite(n), gleicher Filter,
   gleiche Ausstiege) seit 2023: Differenz avgR mit t > 2.
3. **Nutzen im Seitwärtsmarkt**: Portfolio Trend + Range (gemeinsames Konto, max. 20 Pos., ein Coin nur einmal)
   a) im Fenster 15.03.2024 – 31.07.2026 besser als Trend allein (gleiches Risiko 0,95 %), **und**
   b) seit 2023 bei DD ≤ 30 % (Risiko gemeinsam optimiert) höhere Jahresrendite als Trend allein (+34,5 %).
4. **Rückversetz-Test**: Regel nur auf Daten bis 26.04.2026 gewählt, danach 26 Wochen vorwärts: Ergebnis > 0.

Fällt ein Punkt durch: **nicht einbauen**, Ergebnis dokumentieren.

## Ergebnis (25.10.2026) — NICHT BESTANDEN (Kriterium 3b)
- K1/K4: IS-gewählte Regel **Long, Top 2,5 %, mit Trendfilter** — OOS +18,1 %, DD 8,9, 155 Trades, +0,115 R → bestätigt.
- K2: ANN +0,042 R vs. Zufall −0,086 R, t = 5,4 → erfüllt.
- K3a: Seitwärtsfenster 15.03.2024–31.07.2026 bei 0,95 %: Trend −10,7 % (DD 28,5) / Trend+Range −4,7 % (DD 39,3) → knapp erfüllt, aber mit höherem DD.
- K3b: bei DD ≤ 30 % seit 2023: Trend +34,5 %/J, Trend+Range +26,3 %/J, Range allein +13,0 %/J → **nicht erfüllt**.
- Befund: Das Ziel „±1,5 ATR“ führte nicht zu einer Seitwärts-Strategie. Der IS wählte wieder Long im Aufwärtstrend
  (nur mit kürzerem Ziel); Short und Regeln ohne Trendfilter waren im IS schwächer. Zusammen mit der Trend-Regel
  verdünnt das Profil die Kante (+0,103 R statt +0,198 R je Trade). **Nicht eingebaut.**

---

# Vorregistrierung v2 — LETZTER ANLAUF (25.10.2026, vor jeder Rechnung festgelegt)

**Änderung gegenüber v1:** Das Range-Modell darf **nur in Seitwärtsphasen** handeln. v1 erlaubte auch Trendphasen,
und die IS-Wahl nahm genau diese (Long+Trend). v2 testet damit direkt die User-Idee „im Seitwärtsmarkt verdienen“.

- **Seitwärts-Definition (fest, ein Parameter):** auf der Signalkerze |EMA20 − EMA50| < 0,5 × ATR(14).
- Raster: Seiten L/S/LS × Schwelle Top 10/5/2,5/1 % (12 Varianten), immer mit Seitwärts-Bedingung, ohne Trendfilter.
- Gleiche Modelle/Vorhersagen wie v1 (Profil `range`), gleiche Ausstiege (±1,5 ATR, 8 Kerzen), gleiche Kosten.
- IS-Wahl nach avgR (≥ 100 IS-Trades, IS-PnL > 0); Kontrollgruppe: Zufallseinstiege **in denselben Seitwärtskerzen**.
- **Kriterien 1–4 unverändert wie v1.** Fällt einer durch: nicht einbauen, Seitwärts-Thema für den jaegerbot beendet.

## Ergebnis v2 (25.10.2026) — NICHT BESTANDEN → Seitwärts-Thema für den jaegerbot beendet
- IS-Raster: Long in Seitwärtskerzen durchgehend klar negativ (−0,08 bis −0,22 R), LS negativ, nur Short Top 1–2,5 % leicht positiv.
- K1/K4: IS-Wahl Short Top 1 % → OOS −4,2 %, nur 10 Trades, −0,42 R → nicht erfüllt.
- K2: ANN −0,011 R vs. Zufall −0,036 R, t = 0,4 → kein Unterschied zum Zufall, nicht erfüllt.
- K3: Trend+Range seit 2023 +37,2 %/J (vs. 34,5), Seitwärtsfenster +2,3 % (vs. −10,7) — erfüllt, aber ohne Kante
  gegenüber Zufall (K2) und mit negativem OOS; die Verbesserung stammt aus IS-gewählten 169 Trades → als Zufall zu werten.
- Nebenbefund: ANN-**Long**-Signale in Seitwärtskerzen verlieren deutlich. Die Trend-Regel verlangt EMA20 > EMA50,
  handelt aber auch bei sehr kleinem Abstand. Ob ein Mindestabstand die Seitwärts-Verluste der Trend-Regel senkt, wäre
  eine NEUE, aus diesen Daten abgeleitete Hypothese — nur mit eigener Vorregistrierung und Rückversetz-Test.
