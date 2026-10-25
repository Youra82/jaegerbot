# Vorregistrierung: Marktform per reiner Geometrie (25.10.2026)

Festgelegt **bevor** gerechnet wird. User-Idee: „Nutze pure Geometrie, um den Markt zu erkennen.“
Keine Indikatoren (kein EMA/RSI/ADX) für die Erkennung — nur die Form des Kursverlaufs der letzten N = 50 Kerzen (4h).

## Geometrie je Kerze (nur Vergangenheit, Fenster N = 50, inkl. Signalkerze)
- Mittellinie: lineare Regression der Schlusskurse über das Fenster (Steigung m je Kerze).
- Kanal: obere Linie = Mittellinie + max(High − Mittellinie), untere = Mittellinie + min(Low − Mittellinie); Breite W.
- Drift D = |m| × N (Höhenänderung der Mittellinie über das Fenster).
- Geradlinigkeit ER = |Close_t − Close_{t−N}| / Σ|ΔClose| (1 = gerade Linie, 0 = Zickzack).
- Berührungen: Anzahl Kerzen mit High ≥ obere Linie − 0,15·W bzw. Low ≤ untere Linie + 0,15·W, mindestens 3 Kerzen
  Abstand zwischen zwei gezählten Berührungen.
- **Seitwärtskanal** = D < 0,5·W **und** ER < 0,3 **und** ≥ 2 Berührungen oben **und** ≥ 2 unten.

## H1 — Seitwärts handeln (Kanal-Rand)
Nur in Seitwärtskanal-Kerzen: Close ≤ untere Linie + 0,15·W → Long; Close ≥ obere Linie − 0,15·W → Short.
TP = Mittellinie zum Signalzeitpunkt, SL = 0,25·W jenseits der Kanallinie, Zeitausstieg 12 Kerzen; SL vor TP in
derselben Kerze. Kosten 0,22 % je Runde. 39 Coins des Live-Teams, seit 2023-01-01. Keine Parameter-Optimierung.
Kriterien (alle):
1. netto avgR > 0 mit t > 2 seit 2023 und ≥ 60 % der Quartale positiv;
2. Kontrolle: zufällige Einstiege in denselben Seitwärtskanal-Kerzen (zufällige Seite), gleiche Ausstiege: t > 2;
3. OOS 26 W (ab 26.04.2026), 1 % Risiko, max. 20 Pos.: ≥ +10 %, DD ≤ 30 %, ≥ 30 Trades;
4. gemeinsames Konto mit der Trend-Regel: Seitwärtsfenster 15.03.2024–31.07.2026 besser als Trend allein und
   seit 2023 bei DD ≤ 30 % höhere Jahresrendite als Trend allein (+34,5 %).

## H2 — Seitwärtsverluste der Trend-Regel vermeiden
Trend-Signale (Live-Regel 4h, 39 Coins), deren Signalkerze ein Seitwärtskanal ist, werden ausgelassen.
Kriterien (alle):
1. seit 2023 bei DD ≤ 30 % (Risiko neu skaliert) höhere Jahresrendite als ohne Filter (+34,5 %);
2. Seitwärtsfenster 15.03.2024–31.07.2026 besser als ohne Filter (gleiches Risiko 0,95 %);
3. OOS 26 W nicht schlechter als ohne Filter (gleiches Risiko 0,95 %);
4. ausgelassene Trades haben avgR < avgR der behaltenen (Mechanismus-Prüfung).

Je Hypothese gilt: nur einbauen, wenn alle ihre Kriterien erfüllt sind.

## Ergebnis (25.10.2026) — H1 und H2 NICHT BESTANDEN
- 23,7 % aller 4h-Kerzen seit 2023 sind geometrische Seitwärtskanäle.
- **H1 Kanal-Rand handeln:** 13.514 Signale, netto −0,084 R (t −11,1), 2/16 Quartale positiv, OOS −61 %.
  Sogar **schlechter als Zufall** im selben Kanal (−0,059 R, t −3,1): An den Kanalrändern bricht der Kurs öfter aus,
  als er abprallt. Rand-Fading hat auf Krypto-Perps eine negative Kante.
- **H2 Trend-Signale im Kanal auslassen:** Die 220 Trend-Signale aus Seitwärtskanälen sind die **besten**
  (+0,318 R gegenüber +0,169 R der übrigen). Filter senkt die Rendite (+23,1 statt +34,5 %/J), Seitwärtsfenster
  schlechter (−16,3 statt −8,7 %), OOS schlechter (+47,6 statt +51,2 %).
- Befund: Seitwärtsphasen sind nicht der Feind der Trend-Regel — ihre besten Einstiege sind **Ausbrüche aus
  Seitwärtskanälen**. Die Verluste im Seitwärtsfenster kommen aus anderen Signalen.
