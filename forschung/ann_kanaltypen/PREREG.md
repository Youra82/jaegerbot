# Vorregistrierung: ANN lernt Kanaltypen (25.10.2026)

Festgelegt **bevor** trainiert wird. User: „Flache Kanäle = Seitwärtsmarkt … gerne mit ANN anlernen, um die Kanaltypen
zu erkennen.“ Anlass (beschreibend, fibot-Kanal): Trend-Signale bei flachem Kanal +0,36 R (n 80), fallend +0,43 R
(n 27), kein Kanal +0,19 R, steigend +0,01 R (n 89).

## Änderung (Profil `geo`, sonst identisch zur Live-Version `trend`)
12 zusätzliche ANN-Eingaben aus reiner Kursgeometrie, je Fenster N ∈ {50, 150} Kerzen bis einschließlich der
Signalkerze (Mittellinie = Regression der Schlusskurse, Kanal = Hülle der Hochs/Tiefs um die Mittellinie, Breite W):
- `geo_drift_N`  = Steigung × N / W (vorzeichenbehaftet: ≈0 flach, >0 steigend, <0 fallend)
- `geo_er_N`     = Geradlinigkeit |ΔClose über N| / Σ|ΔClose|
- `geo_pos_N`    = Lage des Schlusskurses im Kanal (0 = untere, 1 = obere Linie)
- `geo_width_N`  = W / ATR
- `geo_tu_N`, `geo_tl_N` = Berührungen der oberen/unteren Linie (±0,15·W, ≥ 3 Kerzen Abstand)
Gleiche Labels/Ausstiege (TP 4 ATR vor SL 2 ATR, 12 Kerzen), gleiche Ensembles (3 MLPs je Seite), gleiche
Quartals-Cutoffs ab 2023, gleiche Kosten, gleiche Regel-Suche (IS bis 26.04.2026, OOS danach), gleiches Universum
(39 etablierte Coins), 4h, max. 20 Positionen.

## Kriterien — alle müssen erfüllt sein
1. Regel im OOS bestätigt (wie Optimizer: ≥ +10 %, DD ≤ 30 %, ≥ 10 Trades) und seit 2023 bei DD ≤ 30 % höhere
   Jahresrendite als `trend` (+34,5 %).
2. OOS 26 W bei gleichem Risiko (0,95 %) nicht schlechter als `trend` (+51,2 % im R-Simulator).
3. Seitwärtsfenster 15.03.2024–31.07.2026 bei 0,95 % besser als `trend` (−8,7 %).
4. Kontrollgruppe: ANN-Signale vs. Zufall (gleiche Seite/Filter, gleiche Ausstiege) t > 2.
5. Paritätstest Live-Code = Backtest für alle Configs des neuen Teams.
Fällt ein Punkt durch: nicht einbauen.

---

## Variante B — zusätzlich übergeordnete Kanäle (festgelegt während A trainiert, BEVOR ein Ergebnis von A vorliegt)
User: „Er kann auch gerne übergeordnete Timeframe-Kanäle zur Gesamtentscheidung nutzen.“
Profil `geohtf` = `geo` + 12 weitere Eingaben `htf_geo_*` mit **derselben Geometrie auf Tageskerzen** (N ∈ {50, 150} Tage).
- Tageskerzen aus den 4h-Kerzen gebildet (UTC-Tage); eine 4h-Kerze sieht nur Tage, die zu ihrem Schluss
  **abgeschlossen** sind. Tages-ATR = einfacher 14-Tage-Mittelwert der True Range (exakt gleich live/Backtest).
- Vorlauf 1000 4h-Kerzen (≈ 166 Tage), live werden 1300 Kerzen geladen.
- Alles andere und **alle Kriterien 1–5 wie A**; zusätzlich wird B gegen A berichtet. Eingebaut wird höchstens die
  Variante, die alle Kriterien erfüllt; erfüllen beide, die mit der höheren Jahresrendite bei DD ≤ 30 %.

## Ergebnis A und B (25.10.2026) — beide NICHT BESTANDEN (K1)
- A (`geo`): IS wählte Short+Trend Top 2,5 %, OOS −5,1 % → nicht bestätigt; DD≤30 seit 2023 +7,5 %/J.
- B (`geohtf`): IS wählte **Short im Abwärtstrend Top 10 %**, OOS +66,7 % aber DD 38,3 % > 30 → nicht bestätigt;
  Seitwärtsfenster +548 % (DD 42); DD≤30 seit 2023 +16,3 %/J; Kontrolle +0,062 vs −0,024 R, t 8,2.

## Hypothese C — NACHTRÄGLICH abgeleitet (nach Sicht von B), deshalb nur mit OOS-Maßstab bewertet
Live-Trend-Long (unverändert) + B-Short-Regel (nur im IS gewählt) auf einem Konto, ein Coin nur einmal, max. 20 Pos.,
ein gemeinsamer Risikofaktor. Bestanden nur, wenn **alle**:
1. OOS 26 W: bei dem Risiko, bei dem die Kombination denselben OOS-Drawdown hat wie Trend allein (0,95 %), mehr Ertrag;
2. seit 2023 bei DD ≤ 30 % höhere Jahresrendite als Trend allein (+34,5 %) — wegen IS-Überlappung nur Zusatzbedingung;
3. Seitwärtsfenster bei gemeinsamem 0,95 % besser als Trend allein;
4. Paritätstest Live-Code = Backtest (Profil geohtf) für alle Short-Configs.
Wegen der nachträglichen Ableitung: vor Live-Einsatz zusätzlich mindestens 4 Wochen Vorwärts-Beobachtung (Papier).

## Ergebnis C (25.10.2026) — NICHT BESTANDEN (C1, der OOS-Maßstab)
- C1: Bei gleichem OOS-Drawdown wie Trend allein (6,7 %) bringt die Kombination nur +25,0 % statt +51,2 % → nicht erfüllt.
- C2: seit 2023 bei DD ≤ 30 %: Kombination +51,9 %/J (Faktor 0,55) vs. Trend +34,5 %/J → erfüllt (aber IS-überlappend).
- C3: Seitwärtsfenster +524,9 % (DD 48) vs. −8,7 % → erfüllt.
- Jahre (Trend 0,95 % / Kombi): 2023 +46/+2 %, 2024 +46/+15 %, 2025 −14/+63 %, 2026 +70/+156 %.
- Deutung: Die Short-Regel mit Tageskanälen ist eine echte Kante (t 8,2) und stark genau dann, wenn die Trend-Regel
  stillsteht (2025). Im ungesehenen OOS ist ihr Ertrag je Einheit Drawdown aber schlechter als der der Trend-Regel.
  Nicht live einbauen; Kandidat für eine Papier-Vorwärtsbeobachtung.
