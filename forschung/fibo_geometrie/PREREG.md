# Vorregistrierung: fibot-Geometrie im jaegerbot (25.10.2026)

Festgelegt **bevor** gerechnet wird. User: „pure Geometrie … als Beispiel beim fibot“ → fibot-Geometrie im jaegerbot.
Verwendet wird **der echte fibot-Code** (`fibot/src/fibot/strategy/confluence_logic.py`, `_channel_series`,
`_plan_one_direction`, Parameter DEFAULTS + V2 = strengere Kanaldefinition aus dem fibot-Nachtrag). 4h, die 39 Coins
des Live-Teams, seit 2023-01-01, Kosten 0,22 % je Runde (auch für Limit-Einstiege, konservativ).

## G1 — Markterkennung über fibot-Kanal (Filter der Trend-Regel)
Zustand der Signalkerze j aus dem fibot-Kanal (nur Daten bis j): gültiger Kanal mit Steigung > 0 = „steigend“,
gültiger Kanal mit Steigung ≤ 0 = „flach/fallend“, kein gültiger Kanal = „unklar“.
Regel: Trend-Longs in „flach/fallend“ werden ausgelassen.
Kriterien (alle): (1) seit 2023 bei DD ≤ 30 % höhere Jahresrendite als ohne Filter (+34,5 %); (2) Seitwärtsfenster
15.03.2024–31.07.2026 besser (0,95 %); (3) OOS 26 W nicht schlechter (0,95 %); (4) ausgelassene avgR < behaltene avgR.

## G2 — fibot-Setup als Ort, ANN als Qualität (Seitwärts-Baustein)
fibot-Konfluenz-Fills (Long: Limit an unterer Kanallinie in der 50–61,8-%-Zone des Impulses; Short gespiegelt),
SL/TP wie fibot (SL unter 78,6 % bzw. Linie − 0,2 ATR, TP obere Kanallinie), Sofort-SL in der Fill-Kerze = Verlust,
Zeitausstieg nach 60 Kerzen. **Nur** wenn das jaegerbot-ANN (Trend-Profil, Walk-Forward) auf der letzten geschlossenen
Kerze vor dem Fill in den oberen 10 % liegt (Long: pL ≥ thrL_900, Short: pS ≥ thrS_900).
Kriterien (alle): (1) netto avgR > 0 mit t > 2, ≥ 60 % der Quartale positiv; (2) besser als alle fibot-Fills ohne
ANN-Bedingung (Differenz t > 2); (3) OOS 26 W, 1 % Risiko, max. 20 Pos.: ≥ +10 %, DD ≤ 30 %, ≥ 30 Trades;
(4) gemeinsames Konto mit Trend: Seitwärtsfenster besser als Trend allein und seit 2023 bei DD ≤ 30 % höhere
Jahresrendite als Trend allein.

Je Hypothese: nur einbauen, wenn alle ihre Kriterien erfüllt sind.

**Umsetzungs-Korrektur (nach dem ersten Lauf bemerkt, weil G1 dort 0 Fälle hatte; Regel unverändert):** Der fibot-Kanal der Long-Seite ist
konstruktionsbedingt nur steigend; „flach/fallend“ trat dadurch nie auf (0 Fälle). Fallende Kanäle liefert der fibot
über die gespiegelte Kursreihe (Short-Seite). Zustand jetzt: steigend = Long-Kanal gültig mit Steigung > 0 (und kein
fallender); flach/fallend = gespiegelter Kanal gültig (oder Long-Kanal mit Steigung ≤ 0); sonst unklar.

## Ergebnis (25.10.2026) — G1 und G2 NICHT BESTANDEN
- fibot-Kanalzustand der 4h-Kerzen: unklar 73,5 %, flach/fallend 13,9 %, steigend 12,6 %.
- **G1:** Trend-Signale in flach/fallenden fibot-Kanälen sind die **besten** (n 61, +0,363 R; steigend +0,143 R,
  unklar +0,191 R). Filter senkt die Rendite (+32,9 statt +34,5 %/J), Seitwärtsfenster schlechter (−10,5 statt −8,7 %).
- **G2:** alle fibot-Fills im jaegerbot-Universum −0,221 R (n 293, t −2,3); mit ANN-Bedingung 23 Trades, −0,116 R,
  kein Unterschied (t 0,3); OOS nur 3 Trades; mit Trend zusammen schlechter (+30,5 statt +34,5 %/J).
- Gleicher Befund wie bei der reinen Geometrie (forschung/geometrie): Die stärksten Einstiege der Trend-Regel sind
  Ausbrüche **gegen** eine seitwärts/fallende Struktur; Kauf an der Kanallinie (fibot-Logik) verliert.
