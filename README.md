# JaegerBot — ANN Signal Scoring Trading System

<div align="center">

![JaegerBot](https://img.shields.io/badge/JaegerBot-v3.0-blue?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter--Optimierung-purple?style=for-the-badge)](https://optuna.org/)
[![CCXT](https://img.shields.io/badge/CCXT-Bitget-red?style=for-the-badge)](https://github.com/ccxt/ccxt)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Vollautomatisches KI-Trading-System — ANN-Signale, Walk-Forward-Optimierung, gewichtetes Signal-Scoring, Struktur-basierter SL**

[Übersicht](#übersicht) • [Architektur](#architektur) • [Installation](#installation) • [Konfiguration](#konfiguration) • [Pipeline](#pipeline) • [Live-Trading](#live-trading) • [Monitoring](#monitoring) • [Wartung](#wartung)

</div>

---

## Übersicht

JaegerBot ist ein vollautomatisches Futures-Trading-System für **Bitget**. Das Herzstück ist ein **Artificial Neural Network (ANN)** mit 50 technischen Features, das trainiert wird um direkt die Trade-Outcomes (TP vor SL?) vorherzusagen — nicht nur die Richtung.

Der Bot entscheidet **nicht** binär auf Basis einzelner Indikatoren — stattdessen bewertet ein **gewichtetes Signal-Scoring-System** jeden potenziellen Trade auf einer Skala von 0–10 Punkten. Nur Trades mit ausreichend Gesamtqualität werden eröffnet.

### Kernfunktionen

- **ANN-Modell** (Dense 256→128→64→32→1) mit **50 Features** pro Kerze (inkl. Candle DNA, Pivot-Struktur, Fibonacci-Zonen, Volumen-Richtung)
- **TP/SL-Outcome-Labels** — ANN lernt "trifft TP vor SL?" statt nur "steigt Preis in N Kerzen?"
- **Signal-Scoring mit 6 Komponenten** — inkl. neuer **Struktur-Qualität** (Pivot-Abstand, Fibonacci-Zone, Kerzenstärke)
- **Struktur-basierter Stop-Loss** — SL an Pivot-High/Low + ATR-Puffer statt reinem ATR-Multiplikator
- **Walk-Forward-Validierung (70/30)** im Optimizer — Out-of-Sample-Test verhindert Overfitting
- **Anti-Overtrading-Guard** — Optimizer prunt Configs mit > 150 Trades/Jahr im Test-Set
- **Portfolio-Optimierung** — `./show_results.sh` findet das beste Strategie-Team automatisch
- **Vollständige Analyse-Exports** — `jaegerbot_portfolio_equity.html` + `jaegerbot_trades.xlsx` + Telegram-Versand

### Was ist neu in v2.0 / v3.0

| Komponente | Alt (v1.x) | v2.0 | v3.0 (aktuell) |
|---|---|---|---|
| SuperTrend | Hard-Block bei Gegentrend | Score-Beitrag: 0–2 Punkte | unverändert |
| ADX | Hard-Block bei ADX < 20 | Score-Beitrag: 0–2 Punkte | unverändert |
| Volumen | Hard-Block bei < 80% Avg | Score-Beitrag: 0–1 Punkte | unverändert |
| Volatilität | Hard-Block bei ATR-Spike | Score-Beitrag: 0–1 Punkte | unverändert |
| ANN-Signal | On/Off-Trigger | Score-Beitrag: 0–4 Punkte | 0–3.5 Punkte + min_ann_gate |
| **Struktur-Qualität** | — | — | **neu: 0–1 Punkt** (Pivot-Abstand, Fib-Zone, Körperstärke) |
| ANN-Features | 34 Standard-Indikatoren | 34 | **50** (+ Candle DNA, Pivot, Fibonacci, Vol-Richtung) |
| ANN-Training | Preis-Richtung (N Kerzen) | Preis-Richtung | **TP/SL-Outcome** ("trifft TP vor SL?") |
| Stop-Loss | `initial_sl_pct` (fest %) | ATR-basiert | **Struktur-SL** (Pivot-High/Low + 0.25×ATR) |
| Optimizer DD-Limit | — | 30% | **25%** |
| Optimizer PnL-Min | — | 0% | **5%** |
| Anti-Overtrading | — | — | **neu: max 150 Trades/Jahr** im Test-Set |
| Optimizer-Params | — | 8 Parameter | **13 Parameter** (inkl. max_sl_pct, min_ann_score, structure_weight) |
| Backtester SL | `initial_sl_pct` (fest %) | ATR-basiert | Struktur-basiert |
| Analyse-Export | Nur CSV | HTML + Excel + Telegram | unverändert |

> **Nach dem Update auf v3.0 müssen alle Modelle und Configs neu trainiert werden:**
> ```bash
> rm -rf artifacts/models/ artifacts/db/ artifacts/results/
> rm src/jaegerbot/strategy/configs/config_*.json
> ./run_pipeline.sh
> ```

---

## Signal-Scoring-System

Jeder Trade wird auf einer Skala von 0–10 Punkten bewertet. Ein Trade wird nur eröffnet, wenn der Gesamt-Score `>= min_signal_score`:

```
ANN-Konfidenz     : 0–3.5 Punkte  (Abstand der Vorhersage von der Schwelle)
SuperTrend        : 0–1.5 Punkte  (Trend ausgerichtet = 1.5, Gegentrend = 0)
ADX-Trendstärke   : 0–1.5 Punkte  (ADX >= 35 = 1.5, >= 25 = 1.1, >= 20 = 0.75, >= 15 = 0.375)
Volumen           : 0–1.5 Punkte  (volume_ratio >= 1.2 = 1.5, >= 1.0 = 1.1, >= 0.8 = 0.75)
Volatilität       : 0–1.0 Punkte  (ruhiger Markt = 1.0, ATR-Spike > 2× Avg = 0)
Struktur-Qualität : 0–1.0 Punkte  (Kerzenstärke + Pivot-Abstand + Fibonacci-Zone)
─────────────────────────────────────────────────────────────────────────────────
Gesamt            : 0–10 Punkte → Trade wenn >= min_signal_score (Optimizer-Ergebnis, typ. 6.5–9.0)
```

**Struktur-Qualität (neu in v3.0):**
```
body_to_atr >= 1.5     → +0.3 Pts  (starker Kerzenkörper = klares Signal)
body_to_atr >= 0.8     → +0.15 Pts
dist_to_struct >= 3.0  → +0.3 Pts  (weit von Widerstand = Raum für Bewegung)
dist_to_struct >= 1.5  → +0.15 Pts
in_fib_zone == True    → +0.4 Pts  (Preis in 38–62% Fibonacci-Goldzone)
```

**Minimum ANN-Gate:** Trades werden auch bei hohem Gesamtscore geblockt, wenn das reine ANN-Signal unter `min_ann_score` liegt (Optimizer-Parameter).

**Praxisbeispiel:**
- Starkes ANN-Signal (0.85), ST bullisch, ADX=30, Fib-Zone → Score 7.8 → **Trade** ✓
- Schwaches ANN-Signal (0.61), ADX=8, kein Volumen → Score 0.6 → **kein Trade** ✓ (richtig gefiltert)
- ANN-Score zu niedrig trotz gutem Gesamt-Score → **geblockt** durch min_ann_gate ✓

---

## Architektur

```
OHLCV Marktdaten (500 Kerzen)
        │
        ▼
Feature-Engine (50 Features)
  ├── Volatilität:       BB-Width, BB-Band, ATR (normalisiert), historische Vola
  ├── Momentum:          RSI, MACD, MACD-Diff, Stochastic K/D, Williams %R, ROC, CCI
  ├── Volumen:           OBV, Volume-Ratio, MFI, CMF
  ├── Trend:             ADX, ADX+/-, Price-to-EMA20/50
  ├── Preisstruktur:     High-Low-Range, Close-to-High/Low, Support/Resistance
  ├── Zeit:              Day-of-Week, Hour-of-Day
  ├── Returns:           Lag 1/2/3
  ├── Candle DNA (neu):  body_to_atr, upper/lower_wick_ratio, candle_direction,
  │                      body_midpoint_ratio, bull_streak, bear_streak
  ├── Pivot-Struktur (neu): pivot_high/low_live, dist_to_struct_high/low,
  │                         price_in_range_20, price_in_range_50
  ├── Fibonacci (neu):   fib_position, in_fib_zone (38–62% Goldzone)
  └── Volumen-Richtung (neu): volume_direction, buying_pressure, selling_pressure
        │
        ▼
ANN-Modell (Dense 256→128→64→32→1, Sigmoid)
  ├── BatchNormalization + Dropout nach jeder Schicht (erhöht: 0.4/0.35/0.3/0.25)
  ├── EarlyStopping (patience=15), ReduceLROnPlateau
  ├── Training-Label: "Trifft TP vor SL?" (1=win, 0/−1=loss) — kein Richtungs-Label!
  └── Output: Wahrscheinlichkeit für TP-Hit vor SL-Hit
        │
        ▼
Signal-Scorer (signal_scorer.py)
  ├── ANN-Konfidenz    (0–3.5 Pts)
  ├── SuperTrend       (0–1.5 Pts)
  ├── ADX              (0–1.5 Pts)
  ├── Volumen          (0–1.5 Pts)
  ├── Volatilität      (0–1.0 Pt)
  └── Struktur-Qualität (0–1.0 Pt) ← NEU
        │
  ANN-Score >= min_ann_score?  ← NEU (Gate)
        │
  Score >= min_signal_score?
        │
        ├── NEIN → kein Trade
        └── JA  ──▶ Trade-Eröffnung
                      ├── Leverage + Margin Mode setzen
                      ├── Market-Order platzieren
                      ├── Struktur-basierter SL (Pivot-Low/High + 0.25×ATR) ← NEU
                      └── Trailing Stop (aktiviert bei Profit-Ziel)
```

### Walk-Forward-Validierung (Optimizer)

```
Historische Daten
        │
        ├── 70% Training ──▶ Optuna-Trial (Backtest)
        │                    Prune wenn: DD > 25% oder Trades < 35
        │
        └── 30% Out-of-Sample ──▶ Validierung
                                   Prune wenn: DD > 25% oder PnL ≤ 5%
                                   Prune wenn: Trades/Jahr > 150 (Anti-Overtrading)

Score = log1p(train_pnl) / DD × 0.30
      + log1p(test_pnl)  / DD × 0.70   → Maximierung
```

**Optimizer-Parameter (v3.0):** 13 Parameter werden optimiert:

| Parameter | Bereich | Beschreibung |
|---|---|---|
| `risk_reward_ratio` | 1.5–4.0 | Take-Profit-Multiplikator |
| `risk_per_trade_pct` | 0.5–3.0 | Kapitalrisiko pro Trade |
| `leverage` | 5–20 | Hebel |
| `atr_multiplier_sl` | 1.5–4.0 | ATR-Basis für Struktur-SL |
| `max_sl_pct` | 1.5–4.0 | Maximaler SL in % (Cap für Struktur-SL) |
| `min_sl_pct` | 0.3–1.5 | Minimaler SL in % |
| `trailing_stop_activation_rr` | 0.8–2.5 | RR bei TSL-Aktivierung |
| `trailing_stop_callback_rate_pct` | 0.3–2.0 | TSL Callback-Rate |
| `min_signal_score` | 5.5–9.0 | Score-Schwelle für Trade-Eröffnung |
| `ann_weight` | 2.5–5.0 | Gewicht des ANN-Scores |
| `volume_weight` | 0.5–2.5 | Gewicht des Volumen-Scores |
| `structure_weight` | 0.0–2.0 | Gewicht der Struktur-Qualität |
| `min_ann_score` | 0.5–2.5 | Minimum ANN-Gate (blockiert schwache ANN-Signale) |

### Dateistruktur

```
jaegerbot/
├── src/jaegerbot/
│   ├── strategy/
│   │   ├── run.py                        # Entry-Point pro Strategie
│   │   └── configs/
│   │       └── config_*.json             # Optimierte Konfigurationen (pro Symbol/TF)
│   ├── utils/
│   │   ├── signal_scorer.py              # Signal-Scoring-System (v2.0)
│   │   ├── trade_manager.py              # Live-Trading-Logik, SL/TSL, Order-Platzierung
│   │   ├── ann_model.py                  # Feature-Engineering & ANN-Hilfsfunktionen
│   │   ├── supertrend_indicator.py       # SuperTrend-Implementierung
│   │   ├── exchange.py                   # Bitget CCXT-Wrapper
│   │   └── telegram.py                   # Telegram-Benachrichtigungen
│   └── analysis/
│       ├── backtester.py                 # Backtesting-Engine (ATR-SL, Signal-Scoring)
│       ├── optimizer.py                  # Optuna-Optimierung mit Walk-Forward-Validierung
│       ├── trainer.py                    # ANN-Training
│       ├── find_best_threshold.py        # Threshold-Finder
│       ├── show_results.py               # Analyse, Portfolio-Optimierung, Exports
│       ├── portfolio_simulator.py        # Chronologische Portfolio-Simulation
│       ├── portfolio_optimizer.py        # Greedy-Portfolio-Selektion
│       └── evaluator.py                  # Datensatz-Qualitätsbewertung
├── master_runner.py                      # Haupt-Orchestrator (startet alle aktiven Strategien)
├── auto_optimizer_scheduler.py           # Automatische Reoptimierung nach Zeitplan
├── run_pipeline.sh                       # Vollständiger Pipeline-Run (Train → Optimize)
├── push_configs.sh                       # Optimierte Configs ins Repo pushen
├── show_results.sh                       # Interaktive Analyse & Portfolio-Optimierung
├── show_status.sh                        # Live-Status aller Strategien
├── update.sh                             # VPS-Update (git reset + secret.json sichern)
├── install.sh                            # Erstinstallation
└── settings.json                         # Live-Trading-Konfiguration
```

---

## Installation

### 1. Repository klonen

```bash
git clone https://github.com/Youra82/jaegerbot.git
cd jaegerbot
```

### 2. Virtuelle Umgebung & Abhängigkeiten

```bash
chmod +x install.sh
./install.sh
```

Oder manuell:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. API-Credentials konfigurieren

`secret.json` im Root-Verzeichnis erstellen (wird **nicht** ins Repo committed):

```json
{
  "jaegerbot": [
    {
      "name": "Mein Bitget Account",
      "exchange": "bitget",
      "apiKey": "DEIN_API_KEY",
      "secret": "DEIN_SECRET_KEY",
      "password": "DEIN_PASSPHRASE",
      "options": {
        "defaultType": "swap"
      }
    }
  ],
  "telegram": {
    "bot_token": "DEIN_BOT_TOKEN",
    "chat_id": "DEINE_CHAT_ID"
  }
}
```

> **Wichtig:** Nur Trading-Rechte vergeben, keine Withdrawal-Rechte. IP-Whitelist aktivieren.

---

## Konfiguration

### settings.json

```json
{
  "live_trading_settings": {
    "use_auto_optimizer_results": false,
    "active_strategies": [
      {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "4h",
        "active": true
      },
      {
        "symbol": "ETH/USDT:USDT",
        "timeframe": "6h",
        "active": true
      }
    ]
  },
  "optimization_settings": {
    "enabled": true,
    "num_trials": 500,
    "send_telegram_on_completion": true,
    "constraints": {
      "max_drawdown_pct": 30,
      "min_win_rate_pct": 45,
      "min_pnl_pct": 0
    },
    "schedule": {
      "day_of_week": 6,
      "hour": 23,
      "minute": 0,
      "interval": { "value": 7, "unit": "days" }
    }
  }
}
```

### Config-Dateien (pro Symbol/Timeframe)

Die Configs werden vom Optimizer generiert und in `src/jaegerbot/strategy/configs/` gespeichert:

```json
{
  "_meta": {
    "pnl_pct": 312.45,
    "pnl_pct_oos": 312.45,
    "pnl_pct_train": 489.10,
    "wfv": "70/30"
  },
  "market": {
    "symbol": "BTC/USDT:USDT",
    "timeframe": "4h"
  },
  "strategy": {
    "prediction_threshold": 0.65,
    "min_signal_score": 5.5
  },
  "risk": {
    "margin_mode": "isolated",
    "risk_per_trade_pct": 1.5,
    "risk_reward_ratio": 2.8,
    "leverage": 12,
    "atr_multiplier_sl": 2.8,
    "min_sl_pct": 0.8,
    "trailing_stop_activation_rr": 1.5,
    "trailing_stop_callback_rate_pct": 0.8
  },
  "behavior": {
    "use_longs": true,
    "use_shorts": true
  }
}
```

**Parameter-Erklärung:**

| Parameter | Beschreibung |
|---|---|
| `pnl_pct_oos` | Out-of-Sample PnL (30% Test-Daten) — realistische Erwartung |
| `pnl_pct_train` | Training-PnL (70% Train-Daten) — zum Vergleich |
| `prediction_threshold` | ANN-Schwelle für Long-Signal (Short = 1 − threshold) |
| `min_signal_score` | Mindest-Score 0–10 für Trade-Eröffnung (per Optimizer gefunden) |
| `atr_multiplier_sl` | ATR-Multiplikator für Stop-Loss-Distanz |
| `min_sl_pct` | Minimaler SL in % (Untergrenze für ATR-SL) |
| `trailing_stop_activation_rr` | RR-Vielfaches, bei dem der Trailing Stop aktiviert wird |
| `trailing_stop_callback_rate_pct` | Callback-Rate des Trailing Stops in % |

### Signal-Scoring-Gewichte anpassen

Der Scorer verwendet Standardgewichte (Summe = 10). Diese können per `scoring`-Sektion in der Config überschrieben werden:

```json
{
  "scoring": {
    "ann_weight": 4.0,
    "st_weight": 2.0,
    "adx_weight": 2.0,
    "volume_weight": 1.0,
    "volatility_weight": 1.0
  }
}
```

> In der Regel reicht es, nur `min_signal_score` zu tunen. Die Gewichte werden standardmäßig nicht verändert.

---

## Pipeline

Die Pipeline trainiert das ANN-Modell neu und optimiert alle Parameter für jedes Symbol/Timeframe-Paar.

### Ablauf

```bash
./run_pipeline.sh
```

Der Pipeline-Run umfasst interaktiv konfigurierbare Schritte:

1. **Trainer** → ANN-Modell auf historischen Daten trainieren (Mindest-Accuracy prüfbar)
2. **Threshold-Finder** → optimalen `prediction_threshold` bestimmen
3. **Optimizer** → Optuna optimiert 8 Parameter mit Walk-Forward-Validierung (70/30):
   - `risk_reward_ratio`, `risk_per_trade_pct`, `leverage`
   - `atr_multiplier_sl`, `min_sl_pct`
   - `trailing_stop_activation_rr`, `trailing_stop_callback_rate_pct`
   - `min_signal_score`
4. **Config-Dateien** → werden in `src/jaegerbot/strategy/configs/` gespeichert

**Konfigurierbare Parameter beim Start:**

| Parameter | Beschreibung |
|---|---|
| Symbole | z.B. `BTC ETH XRP SOL` (ohne /USDT) |
| Timeframes | z.B. `15m 1h 4h` (mehrere möglich) |
| Startkapital | Simulationskapital für den Optimizer |
| Trials | Anzahl Optuna-Versuche pro Pair |
| Modus | `strict` (Win-Rate + DD-Filter) oder `best_profit` (nur DD) |
| Startdatum | Historischer Datenzeitraum |

### Optimizer-Modi

```bash
# Strict-Modus: Drawdown < X%, Win-Rate > Y%, PnL > 0 (Standard)
./run_pipeline.sh

# Best-Profit: nur Drawdown begrenzt, maximaler Gewinn
# → Auswahl 2 beim Modus-Dialog
```

### Optimierte Configs pushen

Nach einem erfolgreichen Pipeline-Run die neuen Configs ins Repo:

```bash
./push_configs.sh
```

Das Script staged nur geänderte Config-Dateien, committet mit Zeitstempel und pusht auf `origin/main` (mit automatischem Rebase bei Konflikt).

### Automatischer Optimizer

```bash
# Manuelle Auslösung (sofort)
python auto_optimizer_scheduler.py --force

# Zeitplan-Check (wie der Cron es aufruft)
python auto_optimizer_scheduler.py
```

---

## Live-Trading

### Starten

```bash
# Einzelner Zyklus (für Cron)
source .venv/bin/activate
python master_runner.py

# Dauerbetrieb mit Screen
screen -S jaegerbot
source .venv/bin/activate
python master_runner.py
# Ctrl+A, D zum Detachen
```

### Cron-Setup (empfohlen)

```bash
crontab -e
```

```cron
# Alle 15 Minuten ausführen
*/15 * * * * /usr/bin/flock -n /pfad/zu/jaegerbot/jaegerbot.lock /bin/sh -c "cd /pfad/zu/jaegerbot && .venv/bin/python master_runner.py >> logs/cron.log 2>&1"
```

### Telegram-Benachrichtigungen

Der Bot sendet bei jedem Trade eine Nachricht mit:
- Signal-Score-Breakdown (ANN / ST / ADX / Vol / Vola)
- Entry-Preis, Positionsgröße, Hebel, Margin-Mode
- Stop-Loss (ATR-basiert) und Trailing-Stop-Aktivierungspreis

---

## Monitoring

### Live-Status anzeigen

```bash
./show_status.sh
```

### Portfolio analysieren & Strategien wählen

```bash
./show_results.sh
```

Das Script bietet vier Modi:

| Modus | Beschreibung | Export |
|---|---|---|
| `1` | **Einzel-Analyse** — jede Strategie isoliert backtesten | — |
| `2` | **Manuelle Portfolio-Simulation** — du wählst das Team | HTML + Excel (auf Anfrage) |
| `3` | **Automatische Portfolio-Optimierung** — Bot wählt bestes Team, `settings.json` aktualisierbar | HTML + Excel (automatisch) |
| `4` | **Interaktive Charts** — Entry/Exit-Signale visuell darstellen | — |

Modus 3 generiert automatisch und sendet via Telegram:
- `artifacts/charts/jaegerbot_portfolio_equity.html` — Portfolio-Equity-Kurve mit TP/SL-Markern
- `artifacts/charts/jaegerbot_trades.xlsx` — alle Portfolio-Trades tabellarisch mit Zusammenfassung

### Logs

```bash
# Live-Trading-Log
tail -f logs/cron.log

# Optimizer-Log
tail -f logs/auto_optimizer_trigger.log
```

### Backtest direkt ausführen

```bash
python run_backtest_direct.py
```

---

## Wartung

### VPS-Update

```bash
./update.sh
```

Das Script:
1. Sichert `secret.json`
2. `git fetch && git reset --hard origin/main`
3. Stellt `secret.json` wieder her
4. Bereinigt `.pyc`-Dateien

### Configs nach Update wiederherstellen

Configs werden im Repo versioniert. Nach einem Update:

```bash
git pull origin main
```

Vor einem Update absichern:

```bash
./push_configs.sh
```

### Trade-Lock zurücksetzen

Falls der Bot steckenbleibt und keine neuen Trades öffnet:

```bash
rm artifacts/db/trade_lock.json
```

### Artefakte bereinigen

| Situation | Befehl |
|---|---|
| Autopilot startet zu viele Strategien | `rm artifacts/results/last_optimizer_run.json` |
| Optimizer mit neuer Logik neu starten | `rm artifacts/db/optuna_studies.db` |
| ANN-Modelle neu trainieren (nach Feature-Änderung) | `rm -rf artifacts/models/` |
| Gespeicherte Configs löschen (Pipeline-Ergebnisse) | `rm src/jaegerbot/strategy/configs/config_*.json` |
| Kompletter Neustart aller Artefakte | `rm -rf artifacts/models/ artifacts/db/ artifacts/results/ && rm src/jaegerbot/strategy/configs/config_*.json` |

> **Wichtig:** `artifacts/models/` und `src/jaegerbot/strategy/configs/` sind **zwei getrennte Verzeichnisse**. Modelle löschen entfernt keine Configs — und umgekehrt. Für einen vollständigen Neustart müssen beide gelöscht werden.

> Alternativ beim Pipeline-Start auf `j` antworten — löscht Modelle und Configs automatisch.

---

## Technische Details

### ANN-Modell

- **Architektur:** Dense(256) → Dense(128) → Dense(64) → Dense(32) → Dense(1, Sigmoid)
- **Regularisierung:** BatchNormalization + Dropout(0.4/0.35/0.3/0.25) nach jeder Schicht
- **Features:** **50 Features** (standardisiert mit StandardScaler)
- **Training:** 80/20 Train/Val-Split, EarlyStopping (patience=15), ReduceLROnPlateau
- **Labels:** TP/SL-Outcome — `+1` wenn TP vor SL getroffen, `0/-1` wenn SL zuerst
- **Output:** Wahrscheinlichkeit, dass TP vor SL trifft (0.0–1.0)

**Feature-Gruppen:**

| Gruppe | Features |
|---|---|
| Volatilität | BB-Width, BB-Band, ATR normalisiert, historische Vola |
| Momentum | RSI, MACD, MACD-Diff, Stochastic K/D, Williams %R, ROC, CCI |
| Volumen | OBV, Volume-Ratio, MFI, CMF |
| Trend | ADX, ADX+/-, Price-to-EMA20, Price-to-EMA50 |
| Preisstruktur | High-Low-Range, Close-to-High, Close-to-Low, S/R |
| Zeit | Day-of-Week, Hour-of-Day |
| Returns | Lag 1, Lag 2, Lag 3 |
| **Candle DNA** | body_to_atr, upper_wick_ratio, lower_wick_ratio, candle_direction, body_midpoint_ratio, bull_streak, bear_streak |
| **Pivot-Struktur** | pivot_high_live, pivot_low_live, dist_to_struct_high, dist_to_struct_low, price_in_range_20, price_in_range_50 |
| **Fibonacci** | fib_position, in_fib_zone |
| **Volumen-Richtung** | volume_direction, buying_pressure, selling_pressure |

### Stop-Loss-Berechnung

Der SL ist **struktur-basiert** — er orientiert sich an Pivot-Hochs/Tiefs statt nur am ATR:

```
# Long:
struct_sl    = pivot_low_live − 0.25 × ATR
atr_sl       = entry − ATR × atr_multiplier_sl
sl_distance  = max(entry − struct_sl, entry − atr_sl)
sl_distance  = min(sl_distance, entry × max_sl_pct / 100)   # Cap
sl_distance  = max(sl_distance, entry × min_sl_pct / 100)   # Boden
stop_loss    = entry − sl_distance

# Short: symmetrisch (pivot_high_live + 0.25 × ATR)
```

**Vorteil:** SL liegt unter echter Marktstruktur → weniger unnötige Stop-Outs bei normalem Rauschen.

### Trailing Stop

1. Fixer SL wird sofort nach Entry gesetzt (Sicherheitsnetz)
2. Trailing Stop aktiviert sich bei `entry_price ± sl_distance × activation_rr`
3. Ab Aktivierung: TSL folgt dem Peak-Preis mit `callback_rate_pct` Abstand

### Optuna Walk-Forward-Optimierung

```
Zielfunktion:
  train_score = log1p(train_pnl) / max(train_dd, 0.01)
  test_score  = log1p(test_pnl)  / max(test_dd,  0.01)
  final_score = train_score × 0.30 + test_score × 0.70   → Maximierung

Pruning (strict-Modus):
  - max_drawdown_pct > 25%         → Pruned  (verschärft von 30%)
  - trades_count < 35              → Pruned (Training) / < 15 (Test)
  - test_pnl < 5%                  → Pruned  (verschärft von 0%)
  - test_trades/Jahr > 150         → Pruned  (neu: Anti-Overtrading)
  - win_rate < min_win_rate        → Pruned
```

---

## Systemanforderungen

| Komponente | Minimum | Empfohlen |
|---|---|---|
| CPU | 2 Kerne | 4+ Kerne (Optuna parallel) |
| RAM | 4 GB | 8 GB+ |
| Speicher | 2 GB | 5 GB (Modelle + Daten-Cache) |
| Python | 3.10 | 3.12 |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |

---

## Lizenz

MIT License — siehe [LICENSE](LICENSE)
