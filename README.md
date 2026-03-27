# JaegerBot — ANN Signal Scoring Trading System

<div align="center">

![JaegerBot](https://img.shields.io/badge/JaegerBot-v2.0-blue?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter--Optimierung-purple?style=for-the-badge)](https://optuna.org/)
[![CCXT](https://img.shields.io/badge/CCXT-Bitget-red?style=for-the-badge)](https://github.com/ccxt/ccxt)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Vollautomatisches Deep-Learning Trading-System mit ANN-Signalen und gewichtetem Score-Filter**

[Architektur](#architektur) • [Installation](#installation) • [Konfiguration](#konfiguration) • [Live-Trading](#live-trading) • [Pipeline](#pipeline) • [Monitoring](#monitoring) • [Wartung](#wartung)

</div>

---

## Übersicht

JaegerBot ist ein KI-gesteuertes Futures-Trading-System für Bitget. Das Herzstück ist ein **ANN (Artificial Neural Network)** mit 34 Features, das Trendrichtungen auf Basis historischer Marktdaten prognostiziert. In Version 2.0 ersetzt ein **gewichtetes Signal-Scoring-System** die alte binäre Filter-Kaskade — kein einzelner Indikator kann einen Trade mehr alleine blockieren.

### Was ist neu in v2.0

| Komponente | Alt (v1.x) | Neu (v2.0) |
|---|---|---|
| SuperTrend | Hard-Block bei Gegentrend | Score-Beitrag: 0–2 Punkte |
| ADX | Hard-Block bei ADX < 20 | Score-Beitrag: 0–2 Punkte (graduiert) |
| Volumen | Hard-Block bei < 80% Avg | Score-Beitrag: 0–1 Punkte |
| Volatilität | Hard-Block bei ATR-Spike | Score-Beitrag: 0–1 Punkte |
| ANN-Signal | Reiner On/Off-Trigger | Score-Beitrag: 0–4 Punkte (Konfidenz-gewichtet) |
| Schwelle | Fest (keine Optimierung) | `min_signal_score` per Symbol/Timeframe optimiert |
| Backtester SL | `initial_sl_pct` (fest %) | ATR-basiert (konsistent mit Live-Bot) |

### Signal-Scoring-System

Jeder Trade wird auf einer Skala von 0–10 Punkten bewertet. Ein Trade wird nur eröffnet, wenn der Gesamt-Score `>= min_signal_score`:

```
ANN-Konfidenz   : 0–4 Punkte  (Abstand der Vorhersage von der Schwelle)
SuperTrend      : 0–2 Punkte  (Trend ausgerichtet = 2, Gegentrend = 0)
ADX-Trendstärke : 0–2 Punkte  (ADX >= 35 = 2, >= 25 = 1.5, >= 20 = 1, >= 15 = 0.5)
Volumen         : 0–1 Punkte  (volume_ratio >= 1.2 = 1, >= 1.0 = 0.75, >= 0.8 = 0.5)
Volatilität     : 0–1 Punkte  (ruhiger Markt = 1, Spike > 2× Avg = 0)
─────────────────────────────────
Gesamt          : 0–10 Punkte → Trade wenn >= min_signal_score (z.B. 5.5)
```

**Praxisbeispiel:**
- Starkes ANN-Signal (0.85) im Bärenmarkt, ST bearisch → Score 6.3 → **Trade** (vorher geblockt)
- Schwaches ANN-Signal (0.61), ADX=8, kein Volumen → Score 0.6 → **kein Trade** (richtig gefiltert)

---

## Architektur

```
OHLCV Marktdaten (500 Kerzen)
        │
        ▼
Feature-Engine (34 Features)
  ├── Volatilität: BB-Width, BB-Band, ATR, Keltner, Donchian
  ├── Momentum:    RSI, MACD, Stochastic, Williams %R, ROC, CCI
  ├── Volumen:     OBV, Volume-Ratio, MFI, CMF, VWAP
  ├── Trend:       ADX, ADX+/-, EMA20/50, Price-to-EMA
  ├── Preisstruktur: High-Low-Range, Support/Resistance
  └── Zeit:        Day-of-Week, Hour-of-Day, Lag-Returns
        │
        ▼
ANN-Modell (Dense 256→128→64→32→1, Sigmoid)
  └── Output: 0.0 (stark SHORT) … 1.0 (stark LONG)
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
                      └── Trailing Stop (aktiviert bei Profit)
```

### Dateistruktur

```
jaegerbot/
├── src/jaegerbot/
│   ├── strategy/
│   │   ├── run.py                        # Entry-Point pro Strategie
│   │   └── configs/
│   │       └── config_*.json             # Optimierte Konfigurationen
│   ├── utils/
│   │   ├── signal_scorer.py              # Signal-Scoring-System (v2.0)
│   │   ├── trade_manager.py              # Trading-Logik, Positionsverwaltung
│   │   ├── ann_model.py                  # Feature-Engineering & ANN
│   │   ├── supertrend_indicator.py       # SuperTrend-Implementierung
│   │   ├── exchange.py                   # Bitget CCXT-Wrapper
│   │   └── telegram.py                   # Telegram-Benachrichtigungen
│   └── analysis/
│       ├── backtester.py                 # Backtesting (ATR-SL, Signal-Scoring)
│       ├── optimizer.py                  # Optuna-Optimierung (inkl. min_signal_score)
│       ├── trainer.py                    # ANN-Training
│       ├── find_best_threshold.py        # Threshold-Finder
│       └── show_results.py              # Ergebnisvisualisierung
├── master_runner.py                      # Haupt-Orchestrator
├── auto_optimizer_scheduler.py           # Automatische Reoptimierung
├── push_configs.sh                       # Optimierte Configs pushen
├── run_pipeline.sh                       # Manueller Pipeline-Run
├── update.sh                             # VPS-Update-Script
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
        "use_macd_filter": false,
        "active": true
      },
      {
        "symbol": "ETH/USDT:USDT",
        "timeframe": "6h",
        "use_macd_filter": false,
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
  "_meta": { "pnl_pct": 2616.68 },
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
| `prediction_threshold` | ANN-Schwelle für Long-Signal (Short = 1 - threshold) |
| `min_signal_score` | Mindest-Score 0–10 für Trade-Eröffnung (Optimizer-Ergebnis) |
| `atr_multiplier_sl` | ATR-Multiplikator für Stop-Loss-Distanz |
| `min_sl_pct` | Minimaler SL in % (Untergrenze für ATR-SL) |
| `trailing_stop_activation_rr` | RR-Ratio bei dem der Trailing Stop aktiviert wird |
| `trailing_stop_callback_rate_pct` | Callback-Rate des Trailing Stops in % |

### Signal-Scoring anpassen

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
*/15 * * * * /usr/bin/flock -n /path/to/jaegerbot/jaegerbot.lock /bin/sh -c "cd /path/to/jaegerbot && .venv/bin/python master_runner.py >> logs/cron.log 2>&1"
```

### Telegram-Benachrichtigungen

Der Bot sendet bei jedem Trade eine Nachricht mit:
- Signal-Score-Breakdown (ANN/ST/ADX/Vol/Vola)
- Entry-Preis, Positionsgröße
- Stop-Loss (ATR-basiert) und Trailing-Stop-Aktivierungspreis
- Hebel und Margin-Mode

---

## Pipeline

Die Pipeline trainiert das ANN-Modell neu und optimiert alle Parameter inklusive `min_signal_score`.

### Manueller Run

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Der Pipeline-Run umfasst:
1. **Trainer** → ANN-Modell auf historischen Daten trainieren
2. **Threshold-Finder** → optimalen `prediction_threshold` bestimmen
3. **Optimizer** → Optuna optimiert 8 Parameter:
   - `risk_reward_ratio`, `risk_per_trade_pct`, `leverage`
   - `atr_multiplier_sl`, `min_sl_pct`
   - `trailing_stop_activation_rr`, `trailing_stop_callback_rate_pct`
   - **`min_signal_score`** (neu in v2.0)
4. **Config-Dateien** → werden automatisch überschrieben

### Optimizer-Modus wählen

```bash
# Strict-Modus: Drawdown < 30%, Win-Rate > 45%, PnL > 0
./run_pipeline.sh --mode strict

# Best-Profit: Nur Drawdown begrenzt, maximaler Gewinn
./run_pipeline.sh --mode best_profit
```

### Optimierte Configs pushen

Nach einem erfolgreichen Pipeline-Run die neuen Configs ins Repo:

```bash
./push_configs.sh
```

Das Script:
- Zeigt alle gefundenen Configs mit `min_signal_score` und PnL
- Staged nur geänderte Config-Dateien
- Committet mit Zeitstempel
- Pusht auf `origin/main` (mit automatischem Rebase bei Konflikt)

### Automatischer Optimizer

Der Auto-Optimizer läuft nach Zeitplan (konfigurierbar in `settings.json`):

```bash
# Manuelle Auslösung (sofort)
python auto_optimizer_scheduler.py --force

# Zeitplan-Check (wie der Cron es aufruft)
python auto_optimizer_scheduler.py
```

---

## Monitoring

### Live-Status anzeigen

```bash
./show_status.sh
```

### Logs

```bash
# Live-Trading-Log
tail -f logs/cron.log

# Optimizer-Log
tail -f logs/auto_optimizer_trigger.log
```

### Backtest manuell ausführen

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

Wenn Configs nach einem Update verloren gehen:

```bash
git pull origin main
```

Oder manuell über `push_configs.sh` vor dem Update sichern.

### Trade-Lock zurücksetzen

Falls der Bot steckenbleibt und keine neuen Trades öffnet:

```bash
rm artifacts/db/trade_lock.json
```

### Stale Optimizer-Ergebnisse bereinigen

Falls der Autopilot zu viele Strategien startet:

```bash
rm artifacts/results/last_optimizer_run.json
```

---

## Technische Details

### ANN-Modell

- **Architektur:** Dense(256) → Dense(128) → Dense(64) → Dense(32) → Dense(1, Sigmoid)
- **Regularisierung:** BatchNormalization + Dropout nach jeder Schicht
- **Features:** 34 technische Indikatoren (standardisiert mit StandardScaler)
- **Training:** 80/20 Train/Val-Split, EarlyStopping (patience=15)
- **Output:** 0.0 = stark SHORT, 0.5 = neutral, 1.0 = stark LONG

### Stop-Loss-Berechnung

Der SL ist ATR-basiert und dynamisch:

```
sl_distance = max(ATR × atr_multiplier_sl, entry_price × min_sl_pct)
stop_loss   = entry_price - sl_distance   (Long)
stop_loss   = entry_price + sl_distance   (Short)
```

Gleiche Logik in Backtester und Live-Bot — keine Inkonsistenzen.

### Trailing Stop

1. Fixer SL wird sofort nach Entry gesetzt (Sicherheitsnetz)
2. Trailing Stop aktiviert sich bei `entry_price ± sl_distance × activation_rr`
3. Ab Aktivierung: TSL folgt dem Peak-Preis mit `callback_rate_pct` Abstand

### Optuna-Optimierung

```
Zielfunktion: PnL% / max(Drawdown, 0.01)   → Maximierung
Constraints (strict):
  - max_drawdown_pct < 30%
  - win_rate >= 45%
  - pnl_pct > 0
  - trades_count >= 50
```

---

## Systemanforderungen

| Komponente | Minimum | Empfohlen |
|---|---|---|
| CPU | 2 Kerne | 4+ Kerne (für Optuna parallel) |
| RAM | 4 GB | 8 GB+ |
| Speicher | 2 GB | 5 GB (Modelle + Daten-Cache) |
| Python | 3.10 | 3.11 |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |

---

## Lizenz

MIT License — siehe [LICENSE](LICENSE)
