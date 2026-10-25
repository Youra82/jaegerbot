# Vorregistrierung: Kanal-Varianten für die Short-Regel (25.10.2026, 15:40)

Festgelegt **bevor** trainiert wird. User: „Kannst mit Timeframes spielen, Kanalpräzision, Kanalgrößen usw.“
Ziel: stärkere Kombination Long (Live-Trend-Regel, unverändert) + Short-Komponente (Profil geohtf).

## Basis B0 (bereits trainiert)
4h-Kerzen, übergeordnet Tageskerzen; Kanalfenster (50, 150) auf beiden Ebenen; Berührungstoleranz 0,15·W.

## Varianten — je genau eine Änderung gegenüber B0
| Name | Änderung |
|---|---|
| V1 kurz | Fenster (20, 80) auf 4h und Tag |
| V2 lang | Fenster (100, 300) auf 4h und Tag |
| V3 praezise | Berührungstoleranz 0,08·W |
| V4 grob | Berührungstoleranz 0,25·W |
| V5 woche | übergeordnet Wochenkerzen (Mo-Start), Fenster (20, 50) Wochen |
| V6 basis2h | Basis-Timeframe 2h statt 4h (Labels/Haltedauer wie Profil: 16 Kerzen) |
Alles andere wie B0 (Labels, Ensemble, Cutoffs, Kosten, Regel-Raster, 39 Coins; V6 nur Coins mit 2h-Daten).

## Auswahl — nur In-Sample (2023-01-01 bis 26.04.2026)
Je Variante: Regel im IS gewählt (Optimizer, avgR). Kombination Long + diese Regel (R-Simulator, ein Coin nur einmal,
max. 20 Pos., gemeinsamer Risikofaktor) → **IS-Jahresrendite bei IS-DD ≤ 30 %**. Gewählt wird die beste Variante,
**nur wenn sie B0 im IS schlägt**, sonst bleibt B0.

## Bestätigung — einmalig im OOS (26.04.2026 bis heute), mit dem im IS gewählten Risikofaktor
Gewählte Variante muss in der Kombination im OOS mindestens so viel Ertrag bringen wie B0-Kombination bei ihrem
IS-Faktor, bei OOS-DD ≤ 30 %; und Paritätstest Live = Backtest bestehen. Sonst bleibt B0.

## V7 „Mensch“ (User: „Nutze volle Geometrie wie ein Mensch“ — hinzugefügt vor jedem Ergebnis)
B0 **plus** Chart-Geometrie, wie ein Mensch sie liest, auf 4h (Pivot-Bestätigung k = 5 Kerzen, Rückblick 300 Kerzen)
und Tag (k = 3 Tage, Rückblick 150 Tage, nur abgeschlossene Tage). Ein Pivot zählt erst ab seiner Bestätigung.
- Marktstruktur: letzte zwei bestätigte Hochs/Tiefs → höheres Hoch / höheres Tief / tieferes Hoch / tieferes Tief (Score −2…+2)
- Trendlinien durch die letzten zwei Pivot-Hochs bzw. -Tiefs: Steigung/ATR, Abstand Schlusskurs/ATR, über/unter Linie
- Dreieck/Keil: Konvergenz = Steigung oben − Steigung unten (/ATR)
- Horizontale Zonen: Pivots im Rückblick, gebündelt (±0,5 ATR): nächster Widerstand/Unterstützung — Abstand/ATR und Berührungen
- Doppel-Hoch/Doppel-Tief: letzte zwei Pivot-Hochs bzw. -Tiefs innerhalb 0,5 ATR
- Ausbruch/Retest: Schlusskurs in den letzten 3 Kerzen über Zone mit ≥ 2 Berührungen gestiegen bzw. darunter gefallen;
  Retest = Kurs danach wieder innerhalb 0,5 ATR der gebrochenen Zone
Auswahl/Bestätigung wie alle Varianten.

## Erweiterung (User: „Nutze auch Kerzenformen … lege Bibliotheken an“, „schlauer intelligenter jaegerbot“) — vor jedem Ergebnis
Neues Modul `src/jaegerbot/strategy/candle_patterns.py` (Kerzenformen-Bibliothek, Größen relativ zum ATR, nur
abgeschlossene Kerzen): Hammer, umgekehrter Hammer, Shooting Star, Hängender Mann, Doji, Libellen-/Grabstein-Doji,
Marubozu (bull/bär), Kreisel, Pin-Bar (bull/bär), Engulfing (bull/bär), Harami (bull/bär), Piercing Line,
Dark Cloud Cover, Tweezer Top/Bottom, Inside Bar, Outside Bar, Kicker (bull/bär), Morgenstern, Abendstern,
drei weiße Soldaten, drei schwarze Krähen — auf 4h und auf Tageskerzen (abgeschlossene Tage).
- **V7 „Mensch“** = B0 + Chart-Geometrie (oben) + Kerzenformen.
- **V8 „Kerzen“** = B0 + nur Kerzenformen (Beitrag der Kerzen allein).
Reihenfolge der Trainings wegen Zeit: V7, V8, V5, V1 zuerst; V2, V3, V4, V6 danach, falls bis ca. 19:30 Zeit bleibt.
Nicht gerechnete Varianten zählen als nicht getestet (nicht als abgelehnt).

## „Einheits-ANN“ (User: „Kann man das alles mit der ANN vereinheitlichen?“ — vor jedem Ergebnis hinzugefügt)
E = **ein** V7-Modell (alle 158 Eingaben) für **beide** Richtungen: Long-Regel und Short-Regel werden getrennt aus dem
V7-Modell im IS gewählt (je Seite: Schwelle Top 10/5/2,5/1 % × Trendfilter an/aus, Auswahl nach IS-avgR, ≥ 100 IS-Trades).
Vergleich im IS (Kombination, DD ≤ 30 %) gegen die beste Zwei-Modell-Kombination (Live-Long + Short-Variante).
E wird genommen, wenn es im IS mindestens gleich gut ist (einfacher: ein Modell); OOS einmalig wie oben.

---

## V9 „Mensch mit sauberen Linien“ (25.10.2026, 17:20 — nach User-Kritik an V7-Trendlinien, VOR jedem V9-Ergebnis)
User: „Er findet keine schönen Trendlinien … geraten, nicht intelligent eingezeichnet. Guck, wie der fibot das macht.“
V7 verband die letzten zwei Pivots, auch wenn Schlusskurse die Linie danach brechen. V9 = V7, aber die Trendlinien-
Merkmale (`tlu_*`, `tll_*`, `converg`) werden ersetzt durch Linien nach **fibot-Regeln** (confluence_logic.find_channel),
erweitert auf beide Seiten und jede Steigung:
- Kandidaten: Paare bestätigter Pivots (Tiefs für Unterstützung, Hochs für Widerstand) aus den letzten 8 Pivots im
  Fenster (4h: 150 Kerzen, k = 5; Tag: 150 Tage, k = 3); Abstand ≥ 2k; Linie mindestens 20 Kerzen alt.
- Gültig nur ohne Schlusskurs-Bruch (> 0,5 ATR) seit dem ersten Anker und ohne Pivot jenseits der Linie (> 0,5 ATR);
  ≥ 2 Berührungen (Pivots ≤ 0,3 ATR an der Linie). Auswahl: meiste Berührungen, dann jüngster zweiter Anker.
- Merkmale je Linie: gültig, Steigung/ATR, Abstand Schlusskurs/ATR, Berührungen, Alter; Kanal: Breite/ATR, Lage im Kanal;
  „Linie gebrochen“: Schlusskurs bricht die in der Vorkerze gültige Linie (> 0,5 ATR).
Auswahl wie alle Varianten: IS gegen die Live-Variante V7 („mensch“), nur wenn besser → einmal OOS + Parität.
Wird V9 gewählt und bestätigt, ersetzt es live die Short-Seite; sonst bleibt V7.
- **Nachtrag V9 (nach Sichtprüfung der gefundenen Linien an 4 Beispielen, vor jedem Training/Ergebnis):** eine Linie
  zählt nur, wenn der Schlusskurs höchstens 6 ATR von ihr entfernt ist (Relevanz). Beispiel BTC 19.09.: steiler
  „Widerstand“ durch zwei alte Hochs ~10 ATR über dem Kurs — nie gebrochen, aber für die Lage bedeutungslos.

## Ergebnis V9 (25.10.2026, 17:45) — im IS nicht besser, V7 bleibt
- V7 („mensch“, live): IS-Regel Short Top 1 %, Kombi mit Long +51,4 %/J (DD 29,6).
- V9 (saubere Linien nach fibot-Regeln + Relevanz ≤ 6 ATR): IS-Regel **Long** Top 1 % (keine Short-Seite mehr),
  Kombi mit Long +45,1 %/J (DD 29,9) → schlechter, keine OOS-Prüfung (laut Protokoll).
- Sichtprüfung: V9-Linien sehen menschlicher aus (2–4 echte Berührungen, kein Schlusskurs-Bruch); für die
  Vorhersage der Short-Seite brachten sie aber nichts zusätzlich. Die V7-Merkmale „Linie durch die letzten zwei
  Pivots“ sind als Lesehilfe unschön, beschreiben für das Netz aber offenbar die jüngste Hoch-/Tief-Abfolge.
