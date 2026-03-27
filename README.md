# JaegerBot — ANN Signal Scoring Trading System

<div align="center">

![JaegerBot](https://img.shields.io/badge/JaegerBot-v2.0-blue?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter--Optimierung-purple?style=for-the-badge)](https://optuna.org/)
[![CCXT](https://img.shields.io/badge/CCXT-Bitget-red?style=for-the-badge)](https://github.com/ccxt/ccxt)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Vollautomatisches KI-Trading-System — ANN-Signale, Walk-Forward-Optimierung, gewichtetes Signal-Scoring**

[Übersicht](#übersicht) • [Architektur](#architektur) • [Installation](#installation) • [Konfiguration](#konfiguration) • [Pipeline](#pipeline) • [Live-Trading](#live-trading) • [Monitoring](#monitoring) • [Wartung](#wartung)

</div>

---

## Übersicht

JaegerBot ist ein vollautomatisches Futures-Trading-System für **Bitget**. Das Herzstück ist ein **Artificial Neural Network (ANN)** mit 34 technischen Features, das Kursbewegungen auf Basis historischer OHLCV-Daten prognostiziert.

Der Bot entscheidet **nicht** binär auf Basis einzelner Indikatoren — stattdessen bewertet ein **gewichtetes Signal-Scoring-System** jeden potenziellen Trade auf einer Skala von 0–10 Punkten. Nur Trades mit ausreichend Gesamtqualität werden eröffnet.

### Kernfunktionen

- **ANN-Modell** (Dense 256→128→64→32→1) mit 34 Features pro Kerze
- **Signal-Scoring** statt harter Filter — kein Indikator kann alleine blockieren
- **Walk-Forward-Validierung (70/30)** im Optimizer — Out-of-Sample-Test verhindert Overfitting
- **ATR-basierter Stop-Loss** + **Trailing Stop** — konsistent in Backtest und Live-Bot
- **Portfolio-Optimierung** — `./show_results.sh` findet das beste Strategie-Team automatisch
- **Vollständige Analyse-Exports** — `jaegerbot_portfolio_equity.html` + `jaegerbot_trades.xlsx` + Telegram-Versand

### Was ist neu in v2.0

| Komponente | Alt (v1.x) | Neu (v2.0) |
|---|---|---|
| SuperTrend | Hard-Block bei Gegentrend | Score-Beitrag: 0–2 Punkte |
| ADX | Hard-Block bei ADX < 20 | Score-Beitrag: 0–2 Punkte (graduiert) |
| Volumen | Hard-Block bei < 80% Avg | Score-Beitrag: 0–1 Punkte |
| Volatilität | Hard-Block bei ATR-Spike | Score-Beitrag: 0–1 Punkte |
| ANN-Signal | Reiner On/Off-Trigger | Score-Beitrag: 0–4 Punkte (Konfidenz-gewichtet) |
| Schwelle | Fest | `min_signal_score` per Symbol/Timeframe optimiert |
| Backtester SL | `initial_sl_pct` (fest %) | ATR-basiert (konsistent mit Live-Bot) |
| Optimizer | Nur Training-Daten | Walk-Forward 70/30 (Out-of-Sample-Validierung) |
| Analyse-Export | Nur CSV | HTML-Chart + Excel + CSV via Telegram |

---

## Signal-Scoring-System

Jeder Trade wird auf einer Skala von 0–10 Punkten bewertet. Ein Trade wird nur eröffnet, wenn der Gesamt-Score `>= min_signal_score`:

```
ANN-Konfidenz   : 0–4 Punkte  (Abstand der Vorhersage von der Schwelle)
SuperTrend      : 0–2 Punkte  (Trend ausgerichtet = 2, Gegentrend = 0)
ADX-Trendstärke : 0–2 Punkte  (ADX >= 35 = 2, >= 25 = 1.5, >= 20 = 1, >= 15 = 0.5)
Volumen         : 0–1 Punkte  (volume_ratio >= 1.2 = 1, >= 1.0 = 0.75, >= 0.8 = 0.5)
Volatilität     : 0–1 Punkte  (ruhiger Markt = 1, ATR-Spike > 2× Avg = 0)
─────────────────────────────────────────────────────────────────────────────
Gesamt          : 0–10 Punkte → Trade wenn >= min_signal_score (Optimizer-Ergebnis)
```

**Praxisbeispiel:**
- Starkes ANN-Signal (0.85) im Bärenmarkt, ST bearisch → Score 6.3 → **Trade** ✓ (vorher geblockt)
- Schwaches ANN-Signal (0.61), ADX=8, kein Volumen → Score 0.6 → **kein Trade** ✓ (richtig gefiltert)

---

## Architektur

```
OHLCV Marktdaten (500 Kerzen)
        │
        ▼
Feature-Engine (34 Features)
  ├── Volatilität:    BB-Width, BB-Band, ATR (normalisiert), historische Vola
  ├── Momentum:       RSI, MACD, MACD-Diff, Stochastic K/D, Williams %R, ROC, CCI
  ├── Volumen:        OBV, Volume-Ratio, MFI, CMF
  ├── Trend:          ADX, ADX+/-, Price-to-EMA20/50
  ├── Preisstruktur:  High-Low-Range, Close-to-High/Low, Support/Resistance
  ├── Zeit:           Day-of-Week, Hour-of-Day
  └── Returns:        Lag 1/2/3
        │
        ▼
ANN-Modell (Dense 256→128→64→32→1, Sigmoid)
  ├── BatchNormalization + Dropout nach jeder Schicht
  ├── EarlyStopping (patience=15), ReduceLROnPlateau
  └── Output: 0.0 (stark SHORT) … 0.5 (neutral) … 1.0 (stark LONG)
        │
        ▼
Signal-Scorer (signal_scorer.py)
  ├── ANN-Konfidenz (0–4 Pts)
  ├── SuperTrend    (0–2 Pts)
  ├── ADX           (0–2 Pts)
  ├── Volumen       (0–1 Pt)
  └── Volatilität   (0–1 Pt)
        │
  Score >= min_signal_score?
        │
        ├── NEIN → kein Trade
        └── JA  ──▶ Trade-Eröffnung
                      ├── Leverage + Margin Mode setzen
                      ├── Market-Order platzieren
                      ├── Dynamischer ATR-Stop-Loss
                      └── Trailing Stop (aktiviert bei Profit-Ziel)
```

### Walk-Forward-Validierung (Optimizer)

```
Historische Daten
        │
        ├── 70% Training ──▶ Optuna-Trial (Backtest)
        │                    Prune wenn: DD > 30% oder Trades < 35
        │
        └── 30% Out-of-Sample ──▶ Validierung
                                   Prune wenn: DD > 30% oder PnL ≤ 0

Score = log1p(train_pnl) / DD × 0.30
      + log1p(test_pnl)  / DD × 0.70   → Maximierung
```

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
- **Regularisierung:** BatchNormalization + Dropout nach jeder Schicht
- **Features:** 34 technische Indikatoren (standardisiert mit StandardScaler)
- **Training:** 80/20 Train/Val-Split, EarlyStopping (patience=15), ReduceLROnPlateau
- **Output:** 0.0 = stark SHORT, 0.5 = neutral, 1.0 = stark LONG

### Stop-Loss-Berechnung

Der SL ist ATR-basiert und dynamisch — gleiche Logik in Backtester und Live-Bot:

```
sl_distance = max(ATR × atr_multiplier_sl, entry_price × min_sl_pct)
stop_loss   = entry_price − sl_distance   (Long)
stop_loss   = entry_price + sl_distance   (Short)
```

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
  - max_drawdown_pct > 30%   → Pruned
  - trades_count < 35        → Pruned (Training) / < 15 (Test)
  - test_pnl <= 0            → Pruned
  - win_rate < min_win_rate  → Pruned
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
