# JaegerBot — ANN + Multi-Strategy Kombinations-System

<div align="center">

![JaegerBot](https://img.shields.io/badge/JaegerBot-v4.0-blue?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter--Optimierung-purple?style=for-the-badge)](https://optuna.org/)
[![CCXT](https://img.shields.io/badge/CCXT-Bitget-red?style=for-the-badge)](https://github.com/ccxt/ccxt)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**ANN als Herzstück · EMA-Trendfilter · Fibonacci-Zonenbewertung · Kerzenqualitäts-Gate · Walk-Forward-Optimierung**

[Übersicht](#übersicht) • [Architektur](#architektur) • [Installation](#installation) • [Konfiguration](#konfiguration) • [Pipeline](#pipeline) • [Live-Trading](#live-trading) • [Monitoring](#monitoring) • [Wartung](#wartung)

</div>

---

## Übersicht

JaegerBot ist ein vollautomatisches Futures-Trading-System für **Bitget**. Das Herzstück ist ein **Artificial Neural Network (ANN)** mit 50 technischen Features, das trainiert wird um direkt die Trade-Outcomes vorherzusagen — "trifft TP vor SL?" statt nur Kursrichtung.

Das ANN wird durch drei bewährte Strategie-Konzepte aus dem Bot-Ökosystem ergänzt:

- **EMA Bias Filter** (stbot): Nur Longs in Aufwärtstrends, nur Shorts in Abwärtstrends
- **Fibonacci Golden Zone** (fibot): Einstieg bevorzugt im 38.2%–61.8%-Retracement
- **Candle Body Gate** (vbot): Doji/Indecision-Kerzen werden hart ausgeschlossen

Kein Indikator kann alleine einen Trade erzwingen — alle Komponenten fließen in ein gewichtetes **Signal-Scoring-System** (0–10 Punkte). Nur Trades mit ausreichend Gesamtqualität werden eröffnet.

### Kernfunktionen

- **ANN-Modell** (Dense 256→128→64→32→1) mit **50 Features** pro Kerze
- **TP/SL-Outcome-Labels** — ANN lernt "trifft TP vor SL?" statt Kursrichtung
- **EMA Bias Hard Gate** — kein Long wenn EMA20 < EMA50, kein Short wenn EMA20 > EMA50
- **Candle Body Hard Gate** — `body_to_atr < 0.25` wird sofort abgelehnt
- **Signal-Scoring (6 Komponenten)** — ANN, SuperTrend, ADX, Volumen, Volatilität, Struktur
- **Fibonacci-Gewichtung** — in_fib_zone ist die stärkste Struktur-Komponente (0.40 Anteil)
- **Struktur-basierter Stop-Loss** — SL an Pivot-High/Low + ATR-Puffer
- **Walk-Forward-Validierung (70/30)** — Out-of-Sample-Test verhindert Overfitting
- **Trade-Frequenz-Bonus** — Optimizer bevorzugt aktiv tradende Configs
- **Portfolio-Optimierung** — `./show_results.sh` findet das beste Strategie-Team automatisch
- **Vollständige Analyse-Exports** — Equity-Kurve (HTML) + Trades (Excel) + Telegram

### Neu in v4.0

- **Kombination ANN + stbot + vbot + fibot** als vereinte Strategie
- **EMA Bias Filter** (stbot-inspired): nur trendkonforme Richtung wird gehandelt
- **Candle Body Quality Gate** (vbot-inspired): `body_to_atr < 0.25` → kein Trade
- **Fibonacci-Zone stärker gewichtet**: Anteil im Scorer von 0.10 → 0.40
- **Kritischer Bugfix**: `signal_scorer` normalisiert jetzt `'long'`/`'short'` zu `'buy'`/`'sell'` — ANN-Score und SuperTrend-Check funktionierten davor für LONG-Trades nie korrekt
- **Optimizer-Score-Bugfix**: Drawdown-Einheit (Bruch vs. Prozent) jetzt konsistent (`dd * 100`)
- **Optimizer auf 9 Tunable-Parameter reduziert** (war 13) → bessere Optuna-Abdeckung
- **prediction_threshold jetzt tunable** (0.58–0.80) statt fix
- **Trade-Frequenz-Bonus** + **Win-Rate-Bonus** in Optimizer-Score
- **Anti-Overtrading**: 150 → 300 Trades/Jahr Grenze

> **Nach dem Update auf v4.0 müssen alle Modelle und Configs neu trainiert werden:**
> ```bash
> rm -rf artifacts/models/ artifacts/db/ artifacts/results/
> rm src/jaegerbot/strategy/configs/config_*.json
> ./run_pipeline.sh
> ```

---

## Signal-Scoring-System

Jeder Trade wird auf einer Skala von 0–10 Punkten bewertet. Ein Trade wird nur eröffnet, wenn der Gesamt-Score `>= min_signal_score`. **Vor** dem Scoring greifen zwei harte Gates:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HARD GATES (beide müssen bestehen, sonst sofort skip)                      │
│                                                                             │
│  1. EMA Bias:    Long nur wenn EMA20 > EMA50 (Aufwärtstrend)                │
│                  Short nur wenn EMA20 < EMA50 (Abwärtstrend)                │
│                                                                             │
│  2. Candle Body: body_to_atr >= 0.25 (keine Doji / Indecision-Kerzen)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SIGNAL-SCORING (Soft Gates, gewichtet)                                     │
│                                                                             │
│  ANN-Konfidenz     : 0–3.5 Punkte  (Abstand der Vorhersage von Schwelle)   │
│  SuperTrend        : 0–1.5 Punkte  (Trendausrichtung, kein Hard-Block)      │
│  ADX-Trendstärke   : 0–1.5 Punkte  (>= 35 = voll, >= 25 = 75%, ...)        │
│  Volumen           : 0–1.5 Punkte  (volume_ratio >= 1.2 = voll, ...)        │
│  Volatilität       : 0–1.0 Punkte  (ruhiger Markt = voll, Spike = 0)       │
│  Struktur-Qualität : 0–1.0 Punkte  (Kerze + Pivot-Abstand + Fibonacci)     │
│  ─────────────────────────────────────────────────────                      │
│  Gesamt            : 0–10 Punkte → Trade wenn >= min_signal_score           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Struktur-Qualität (Budgetaufteilung):**
```
Kerzenqualität (vbot)  : body_to_atr >= 0.5 + clean wick  → 0.30 × structure_weight
Pivot-Abstand          : dist_to_struct >= 2.0 ATR         → 0.30 × structure_weight
Fibonacci Golden Zone  : in_fib_zone (38.2%–61.8%)        → 0.40 × structure_weight  ← stärkstes Signal
```

**Minimum ANN-Gate:** Trades werden auch bei hohem Gesamt-Score geblockt, wenn das reine ANN-Signal unter `min_ann_score` liegt (fest: 1.5 Punkte).

---

## Architektur

```
OHLCV Marktdaten (500 Kerzen)
        │
        ▼
Feature-Engine (50 Features — ann_model.py)
  ├── Volatilität:        BB-Width, BB-Band, ATR (norm.), historische Vola
  ├── Momentum:           RSI, MACD, MACD-Diff, Stochastic K/D, Williams %R, ROC, CCI
  ├── Volumen:            OBV, Volume-Ratio, MFI, CMF
  ├── Trend:              ADX, ADX+/-, Price-to-EMA20/50
  ├── Preisstruktur:      High-Low-Range, Close-to-High/Low, Support/Resistance
  ├── Zeit:               Day-of-Week, Hour-of-Day
  ├── Returns:            Lag 1/2/3, historische Vola
  ├── Candle DNA:         body_to_atr, upper/lower_wick_ratio, candle_direction,
  │                       body_midpoint_ratio, bull_streak, bear_streak
  ├── Pivot-Struktur:     dist_to_struct_high/low, price_in_range_20/50
  ├── Fibonacci:          fib_position, in_fib_zone (38–62% Goldzone)
  └── Volumen-Richtung:   volume_direction, buying_pressure, selling_pressure
        │
        ▼
ANN-Modell (ann_model.py)
  ├── Dense(256) → BatchNorm → Dropout(0.4)
  ├── Dense(128) → BatchNorm → Dropout(0.35)
  ├── Dense(64)  → Dropout(0.3)
  ├── Dense(32)  → Dropout(0.25)
  ├── Dense(1, Sigmoid)
  ├── Label: "Trifft TP vor SL?" (1 = TP zuerst, 0 = SL zuerst)
  └── Output: Wahrscheinlichkeit 0.0–1.0
        │
        ▼
Hard Gate 1: EMA Bias Filter (stbot-inspired)
  ├── Long signal:  EMA20 > EMA50? → weiter / sonst skip
  └── Short signal: EMA20 < EMA50? → weiter / sonst skip
        │
        ▼
Hard Gate 2: Candle Body Quality (vbot-inspired)
  └── body_to_atr >= 0.25? → weiter / sonst skip
        │
        ▼
Signal-Scorer (signal_scorer.py)
  ├── ANN-Konfidenz    (0–3.5 Pts)
  ├── SuperTrend       (0–1.5 Pts)
  ├── ADX              (0–1.5 Pts)
  ├── Volumen          (0–1.5 Pts)
  ├── Volatilität      (0–1.0 Pt)
  └── Struktur-Qualität (0–1.0 Pt) — Fibonacci 40%, Kerze 30%, Pivot 30%
        │
  ANN-Score >= min_ann_score (1.5)?
        │
  Gesamt-Score >= min_signal_score?
        │
        ├── NEIN → kein Trade
        └── JA  ──▶ Trade-Eröffnung
                      ├── Leverage + Margin Mode setzen
                      ├── Market-Order platzieren
                      ├── Struktur-basierter SL (Pivot-Low/High + 0.25×ATR)
                      └── Trailing Stop (aktiviert bei Profit-Ziel)
```

### Walk-Forward-Validierung (Optimizer)

```
Historische Daten
        │
        ├── 70% Training ──▶ Optuna-Trial
        │                    Prune wenn: DD > 25% oder Trades < 35
        │
        └── 30% Out-of-Sample ──▶ Validierung
                                   Prune wenn: DD > 25%, PnL <= 0, Trades < 5
                                   Prune wenn: Trades/Jahr > 300 (Anti-Overtrading)
                                   Strict-Modus: Win-Rate < min_wr oder PnL < min_pnl

Score = log1p(train_pnl%) / max(train_dd×100, 1.0) × 0.30
      + log1p(test_pnl%)  / max(test_dd×100,  1.0) × 0.70
      + log1p(test_trades) × 2.0                       (Trade-Frequenz-Bonus)
      + max(0, (test_winrate − 40) / 10)               (Win-Rate-Bonus)
```

**Optimizer-Parameter (v4.0) — 9 tunable, 4 fixed:**

| Parameter | Bereich | Typ |
|---|---|---|
| `prediction_threshold` | 0.58–0.80 | tunable |
| `risk_reward_ratio` | 1.5–8.0 | tunable |
| `risk_per_trade_pct` | 1.0–5.0 | tunable |
| `leverage` | 10–50 | tunable |
| `atr_multiplier_sl` | 1.0–4.0 | tunable |
| `max_sl_pct` | 0.8–3.0 | tunable |
| `trailing_stop_activation_rr` | 0.8–3.0 | tunable |
| `trailing_stop_callback_rate_pct` | 0.3–2.0 | tunable |
| `min_signal_score` | 5.0–9.0 | tunable |
| `min_sl_pct` | 0.3 | fest |
| `min_ann_score` | 1.5 | fest |
| `ann_weight` | 3.5 | fest |
| `volume_weight` | 1.5 | fest |
| `structure_weight` | 1.0 | fest |

### Dateistruktur

```
jaegerbot/
├── src/jaegerbot/
│   ├── strategy/
│   │   ├── run.py                        # Entry-Point pro Strategie
│   │   └── configs/
│   │       └── config_*.json             # Optimierte Konfigurationen (pro Symbol/TF)
│   ├── utils/
│   │   ├── signal_scorer.py              # Signal-Scoring-System (6 Komponenten)
│   │   ├── trade_manager.py              # Live-Trading, EMA/Body-Filter, SL/TSL
│   │   ├── ann_model.py                  # Feature-Engineering (50 Features) & ANN
│   │   ├── supertrend_indicator.py       # SuperTrend-Implementierung
│   │   ├── exchange.py                   # Bitget CCXT-Wrapper
│   │   └── telegram.py                   # Telegram-Benachrichtigungen
│   └── analysis/
│       ├── backtester.py                 # Backtesting-Engine (EMA/Body-Filter, Signal-Scoring)
│       ├── optimizer.py                  # Optuna-Optimierung mit Walk-Forward-Validierung
│       ├── trainer.py                    # ANN-Training
│       ├── find_best_threshold.py        # Threshold-Finder
│       ├── show_results.py               # Analyse, Portfolio-Optimierung, Exports
│       ├── portfolio_simulator.py        # Chronologische Portfolio-Simulation
│       ├── portfolio_optimizer.py        # Greedy-Portfolio-Selektion
│       └── evaluator.py                  # Datensatz-Qualitätsbewertung
├── master_runner.py                      # Haupt-Orchestrator
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
    "num_trials": 200,
    "send_telegram_on_completion": true,
    "constraints": {
      "max_drawdown_pct": 25,
      "min_win_rate_pct": 55,
      "min_pnl_pct": 5
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
    "prediction_threshold": 0.68,
    "min_signal_score": 6.5
  },
  "risk": {
    "margin_mode": "isolated",
    "risk_per_trade_pct": 2.0,
    "risk_reward_ratio": 3.0,
    "leverage": 15,
    "atr_multiplier_sl": 2.5,
    "min_sl_pct": 0.3,
    "max_sl_pct": 2.0,
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
| `prediction_threshold` | ANN-Schwelle für Long-Signal (Short = 1 − threshold); vom Optimizer gefunden |
| `min_signal_score` | Mindest-Score 0–10 für Trade-Eröffnung (vom Optimizer gefunden) |
| `atr_multiplier_sl` | ATR-Multiplikator für Stop-Loss-Distanz |
| `min_sl_pct` | Minimaler SL in % (Untergrenze) |
| `max_sl_pct` | Maximaler SL in % (Cap für Struktur-SL) |
| `trailing_stop_activation_rr` | RR-Vielfaches, bei dem Trailing Stop aktiviert wird |
| `trailing_stop_callback_rate_pct` | Callback-Rate des Trailing Stops in % |

---

## Pipeline

### Ablauf

```bash
./run_pipeline.sh
```

Der Pipeline-Run umfasst drei Stufen pro Symbol/Timeframe-Paar:

1. **Trainer** → ANN-Modell auf historischen Daten trainieren (Mindest-Accuracy konfigurierbar)
2. **Threshold-Finder** → optimalen `prediction_threshold` bestimmen
3. **Optimizer** → Optuna optimiert 9 Parameter mit Walk-Forward-Validierung (70/30):
   - `prediction_threshold`, `risk_reward_ratio`, `risk_per_trade_pct`, `leverage`
   - `atr_multiplier_sl`, `max_sl_pct`
   - `trailing_stop_activation_rr`, `trailing_stop_callback_rate_pct`
   - `min_signal_score`
4. **Config-Dateien** → in `src/jaegerbot/strategy/configs/` gespeichert

Bei Misserfolg (Accuracy zu niedrig oder kein gültiger Threshold) werden bis zu 3 verschiedene Datenzeiträume automatisch versucht.

**Konfigurierbare Parameter beim Start:**

| Parameter | Beschreibung |
|---|---|
| Symbole | z.B. `BTC ETH XRP SOL` (ohne /USDT) |
| Timeframes | z.B. `1h 4h 6h 1d` |
| Startkapital | Simulationskapital für den Optimizer |
| Trials | Anzahl Optuna-Versuche pro Pair (Standard: 200) |
| Modus | `strict` (Win-Rate + DD-Filter) oder `best_profit` (nur DD) |
| Startdatum | Historischer Datenzeitraum (oder `a` für Automatik) |

**Empfohlene Zeitfenster:**

| Timeframe | Empfohlener Rückblick |
|---|---|
| 5m, 15m | 180–365 Tage |
| 30m, 1h | 180–365 Tage |
| 2h, 4h | 550–730 Tage |
| 6h, 1d | 1095–1825 Tage |

### Optimizer-Modi

```bash
# Strict-Modus: Drawdown < X%, Win-Rate > Y%, PnL > Z% (Standard — empfohlen)
./run_pipeline.sh
# → Auswahl 1 beim Modus-Dialog

# Best-Profit: nur Drawdown begrenzt, maximaler Gewinn
# → Auswahl 2 beim Modus-Dialog
```

### Optimierte Configs pushen

```bash
./push_configs.sh
```

Das Script staged nur geänderte Config-Dateien, committet mit Zeitstempel und pusht auf `origin/main`.

### Automatischer Optimizer

```bash
# Manuelle Auslösung (sofort)
./auto_optimizer_scheduler.py --force

# Zeitplan-Check (wie der Cron es aufruft)
./auto_optimizer_scheduler.py
```

---

## Live-Trading

### Starten

```bash
# Einzelner Zyklus (für Cron)
source .venv/bin/activate
python3 master_runner.py

# Dauerbetrieb mit Screen
screen -S jaegerbot
source .venv/bin/activate
python3 master_runner.py
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

### Signal-Ablauf im Live-Betrieb

Bei jeder Ausführung prüft `trade_manager.py` für jede aktive Strategie:

1. ANN-Vorhersage berechnen (letzte abgeschlossene Kerze)
2. **EMA Bias Check**: EMA20 vs. EMA50 — Gegentrend → skip
3. **Candle Body Check**: body_to_atr >= 0.25 — Doji → skip
4. **Signal-Scoring**: 6 Komponenten, Score >= min_signal_score?
5. **ANN-Gate**: ANN-Score >= min_ann_score?
6. Trade eröffnen: Market-Order → fixer SL → Trailing Stop

### Telegram-Benachrichtigungen

Der Bot sendet bei jedem Trade:
- Signal-Score-Breakdown (ANN / ST / ADX / Vol / Vola / Struktur)
- Entry-Preis, Positionsgröße, Hebel, Margin-Mode
- Stop-Loss (ATR-basiert / Struktur) und Trailing-Stop-Aktivierungspreis

---

## Monitoring

### Tests ausführen

```bash
./run_tests.sh
```

Führt alle Pytest-Tests aus (Live-Workflow, Trailing-Stop, Order-Placement). Gibt `3 passed, 3 skipped` bei korrekter Installation zurück.

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
| `1` | **Einzel-Analyse** — jede Strategie isoliert backtesten inkl. Win-Rate | — |
| `2` | **Manuelle Portfolio-Simulation** — du wählst das Strategie-Team | HTML + Excel |
| `3` | **Automatische Portfolio-Optimierung** — Bot wählt bestes Team, `settings.json` aktualisierbar | HTML + Excel |
| `4` | **Interaktive Charts** — Entry/Exit-Signale visuell darstellen | — |

Modus 3 generiert automatisch:
- `artifacts/charts/jaegerbot_portfolio_equity.html` — Portfolio-Equity-Kurve
- `artifacts/charts/jaegerbot_trades.xlsx` — alle Portfolio-Trades tabellarisch

### Logs

```bash
# Live-Trading-Log
tail -f logs/cron.log

# Optimizer-Log
tail -f logs/auto_optimizer_trigger.log
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

Configs werden im Repo versioniert:

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
| Gespeicherte Configs löschen | `rm src/jaegerbot/strategy/configs/config_*.json` |
| Kompletter Neustart | `rm -rf artifacts/models/ artifacts/db/ artifacts/results/ && rm src/jaegerbot/strategy/configs/config_*.json` |

> **Wichtig:** `artifacts/models/` und `src/jaegerbot/strategy/configs/` sind zwei getrennte Verzeichnisse. Für einen vollständigen Neustart (z.B. nach Feature-Änderungen) müssen beide gelöscht werden — oder beim Pipeline-Start `j` eingeben.

---

## Technische Details

### ANN-Modell

- **Architektur:** Dense(256) → Dense(128) → Dense(64) → Dense(32) → Dense(1, Sigmoid)
- **Regularisierung:** BatchNormalization + Dropout(0.4/0.35/0.3/0.25) nach jeder Schicht
- **Features:** 50 Features (standardisiert mit StandardScaler / RobustScaler)
- **Training:** 80/20 Train/Val-Split, EarlyStopping (patience=15), ReduceLROnPlateau
- **Labels:** TP/SL-Outcome — `+1` wenn TP vor SL getroffen, `−1` wenn SL zuerst
- **Output:** Wahrscheinlichkeit, dass TP vor SL trifft (0.0–1.0)

**Feature-Gruppen:**

| Gruppe | Features |
|---|---|
| Volatilität | BB-Width, BB-Band, ATR norm., historische Vola |
| Momentum | RSI, MACD, MACD-Diff, Stochastic K/D, Williams %R, ROC, CCI |
| Volumen | OBV, Volume-Ratio, MFI, CMF |
| Trend | ADX, ADX+/-, Price-to-EMA20, Price-to-EMA50 |
| Preisstruktur | High-Low-Range, Close-to-High, Close-to-Low, S/R |
| Zeit | Day-of-Week, Hour-of-Day |
| Returns | Lag 1, Lag 2, Lag 3 |
| **Candle DNA** | body_to_atr, upper_wick_ratio, lower_wick_ratio, candle_direction, body_midpoint_ratio, bull_streak, bear_streak |
| **Pivot-Struktur** | dist_to_struct_high/low, price_in_range_20/50 |
| **Fibonacci** | fib_position, in_fib_zone (38.2%–61.8% Goldzone) |
| **Volumen-Richtung** | volume_direction, buying_pressure, selling_pressure |

### Stop-Loss-Berechnung

Der SL ist **struktur-basiert** — er orientiert sich an Pivot-Hochs/Tiefs:

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

### Trailing Stop

1. Fixer SL wird sofort nach Entry gesetzt (Sicherheitsnetz)
2. Trailing Stop aktiviert sich bei `entry ± sl_distance × activation_rr`
3. Ab Aktivierung: TSL folgt dem Peak-Preis mit `callback_rate_pct` Abstand

### EMA Bias Filter (stbot-inspired)

```python
# Aufwärtstrend (EMA20 > EMA50) → nur Longs
# Abwärtstrend (EMA20 < EMA50) → nur Shorts
# Neutraler Markt (EMA20 ≈ EMA50) → kein Trade
```

Verhindert Gegentrend-Trades — vergleichbar mit dem MTF-EMA-Bias im stbot, der als wichtigster Win-Rate-Treiber identifiziert wurde.

### Candle Body Gate (vbot-inspired)

```python
# body_to_atr = abs(close - open) / ATR
# < 0.25 → Doji / Indecision-Kerze → kein Trade
# >= 0.25 → klares Momentum → Trade erlaubt
```

Sorgt dafür, dass nur an Kerzen mit echtem Momentum getradet wird.

### Fibonacci Golden Zone (fibot-inspired)

```python
# 10-Kerzen Swing High/Low
# in_fib_zone = True wenn close im Bereich 38.2%–61.8% des Swings
# Anteil im Struktur-Score: 0.40 (stärkste Einzel-Komponente)
```

Die Golden Zone (38.2%–61.8%) ist ein Retracement-Niveau, an dem im fibot konsistent gute Einstiege entstanden.

### Optimizer-Score-Formel

```python
train_score = log1p(max(0, train_pnl)) / max(train_dd * 100, 1.0)
test_score  = log1p(max(0, test_pnl))  / max(test_dd  * 100, 1.0)
trade_bonus = log1p(test_trades) * 2.0          # aktive Configs bevorzugen
wr_bonus    = max(0, (test_winrate - 40) / 10)  # Win-Rate > 40% belohnen

final_score = train_score * 0.30 + test_score * 0.70 + trade_bonus + wr_bonus
```

`test_dd` ist ein Bruch (0.09 = 9%) → `× 100` für konsistente Einheiten mit `test_pnl` (%).

---

## Coin & Timeframe Empfehlungen

Die Eignung eines Coin/Timeframe-Paares ergibt sich direkt aus den **effektiven Zeitspannen** der internen Indikatoren. Jedes Fenster (EMA, Fibonacci, Pivot-SL) multipliziert sich mit dem Timeframe — zu kurze Zeitspannen erzeugen Noise statt Struktur.

### Effektive Zeitspannen je Timeframe

| TF | EMA20 | EMA50 | Fibonacci (10K) | Pivot-SL (20K) | Lookahead | Geeignet |
|---|---|---|---|---|---|---|
| 5m | 1.7h | 4.2h | 50 min | 1.7h | 1h | ❌ |
| 15m | 5h | 12.5h | 2.5h | 5h | 3h | ⚠️ |
| 30m | 10h | 25h | 5h | 10h | 6h | ⚠️ |
| 1h | 20h | 50h | 10h | 20h | 8h | ✅ |
| 2h | 40h | 100h | 20h | 40h | 10h | ✅ |
| **4h** | **3.3d** | **8.3d** | **1.7d** | **3.3d** | **20h** | **✅✅** |
| **6h** | **5d** | **12.5d** | **2.5d** | **5d** | **24h** | **✅✅** |
| 1d | 20d | 50d | 10d | 20d | 5d | ✅ |

**Warum 5m/15m nicht funktionieren:**
- Fibonacci über 50–150 Minuten hat keine technische Aussagekraft
- EMA20/50 kreuzt mehrmals täglich → EMA-Bias-Filter blockiert fast alle Signale
- Anti-Overtrading-Guard (300 Trades/Jahr) greift bei 5m systematisch: mit ~77.000 Kerzen/Jahr werden weit über 300 Trades generiert → alle Trials gepruned

**Warum 4h/6h optimal sind:**
- EMA20/50 spannt 3–12 Tage → echter Trendfilter, nicht täglich wechselnd
- Fibonacci über 1.7–2.5 Tage → echte Swing-Retracements erkennbar
- Pivot-SL über 3–5 Tage → strukturell valide Hochs/Tiefs
- Historisch beobachtete ANN-Accuracy: **75%+ bei 6h**, **80%+ bei 1d**
- Ausreichend Kerzen (730–1095 Tage Lookback) für qualitativ hochwertiges Training

### Trainingsdaten pro Lookback (Pipeline-Defaults)

| TF | Lookback | Kerzen gesamt | ANN-Signale (~68%) | Min Train Trades |
|---|---|---|---|---|
| 5m | 270d | 77.760 | ~52.900 | 272 ⚠️ Anti-OT |
| 15m | 270d | 25.920 | ~17.600 | 90 |
| 30m | 365d | 17.520 | ~11.900 | 61 |
| 1h | 365d | 8.760 | ~5.950 | 30 |
| 4h | 730d | 4.380 | ~2.980 | 15 |
| 6h | 1095d | 4.380 | ~2.980 | 15 |
| 1d | 1095d | 1.095 | ~745 | 3 ⚠️ zu wenig |

### Coin-Eignung

| Coin | Trend-Qualität | Fibonacci-Verhalten | ANN-Lernbarkeit | Bewertung |
|---|---|---|---|---|
| **BTC** | Starke, lange Trends | Institutionelle S/R exakt an Fib-Levels | Sauberste Daten, konsistenteste Muster | ✅✅ Beste Wahl |
| **ETH** | Ähnlich BTC, etwas mehr Volatilität | Gute Fib-Struktur | Sehr gute Datenbasis | ✅✅ Sehr gut |
| **SOL** | Starke Trends, klare Breakouts | Klare Swings | Gute Liquidität | ✅ Gut |
| **BNB** | Stabiler Trend, niedrige Volatilität | Moderate Swings | Zuverlässige Daten | ✅ Gut |
| **LTC** | Folgt BTC-Muster eng, stabile Trends | Sehr gute Fib-Struktur (BTC-korreliert) | Hohe Liquidität, lange Historie | ✅ Gut |
| **AVAX** | Starke Trends in Bullphasen, klare Swings | Gute Fib-Levels, institutionell gehandelt | Ausreichend Liquidität und Daten | ✅ Gut |
| **TON** | Wachsendes Ecosystem, stabile Aufwärtstrends | Moderate Fib-Struktur | Zunehmende Liquidität | ✅ Gut |
| **INJ** | Explosive Trends, starke Direktionalität | Klare Swings in Trendbewegungen | Hohes Volumen in Bullphasen | ✅ Gut |
| **ARB** | ETH-korreliert, gute Trendbewegungen | Vernünftige Fib-Struktur | Gute Datenbasis seit 2023 | ✅ Gut |
| **MATIC/POL** | Gute Trends in Bullphasen, rangelastig in Bear | Moderate Fib-Swings | Hohe Liquidität, lange Historie | ✅ Gut |
| **XRP** | Kann monatelang seitwärts laufen | Moderat | Phasenabhängig | ⚠️ Mittel |
| **LINK** | Starke Bullphasen, aber oft rangelastig | Moderat | Mittel | ⚠️ Mittel |
| **DOT** | Lange Seitwärtsphasen, selten klare Trends | Schwache Fib-Gültigkeit in Seitwärtsmärkten | Unzuverlässig in Bear | ⚠️ Mittel |
| **SUI** | Starke Trending-Phasen, aber junge Daten | Gute Swings wenn trending | Wenig historische Daten (ab 2023) | ⚠️ Mittel |
| **NEAR** | Moderate Trends, häufige Konsolidierung | Schwache Fib-Struktur | Mittelmäßige Liquidität | ⚠️ Mittel |
| **UNI** | DeFi-getrieben, trendabhängig | Moderat | Phasenweise gute Muster | ⚠️ Mittel |
| **ADA** | Sehr lange Seitwärtsphasen | Schwache Swings | Schlecht in Bear | ⚠️ Schwach |
| **DOGE** | Sentiment-getrieben, kaum Struktur | Geringe Fib-Gültigkeit | Hohe Noise-Rate | ❌ Schlecht |
| **SHIB/PEPE** | Reine Pump-Coins | Keine Fib-Struktur | Nicht lernbar | ❌❌ Nicht geeignet |

### Empfohlene Kombinationen (Ranking)

| Rang | Kombination | Erwartete ANN-Accuracy | Begründung |
|---|---|---|---|
| 🥇 1 | **BTC 4h** | 75–85% | Beste Trendklarheit + Fibonacci + 730 Tage Daten |
| 🥇 1 | **BTC 6h** | 75–85% | Noch sauberere Signale, weniger Rauschen |
| 🥈 2 | **ETH 4h** | 72–82% | Ähnlich BTC, etwas mehr Bewegung |
| 🥈 2 | **ETH 6h** | 72–82% | Ideal für höhere Accuracy |
| 🥉 3 | **BTC 1h** | 68–75% | Mehr Trades, aber mehr Noise als 4h |
| 4 | **SOL 4h** | 65–75% | Gute Trends, aber volatiler |
| 4 | **BNB 4h** | 65–72% | Stabil, aber weniger Bewegung |
| 4 | **LTC 4h / 6h** | 65–75% | BTC-Muster, hohe Liquidität |
| 4 | **AVAX 4h** | 65–73% | Klare Bulltrends, gute Fib-Levels |
| 5 | **ARB 4h** | 62–72% | ETH-korreliert, gute Trendbewegungen |
| 5 | **MATIC 4h** | 62–70% | Gut in Bullphasen, sonst rangelastig |
| 5 | **INJ 4h** | 62–72% | Explosiv in Trends, aber volatiler |
| 5 | **TON 4h** | 60–70% | Wachsende Datenbasis, stabile Trends |
| 5 | **BTC 1d** | 80–88% | Beste Accuracy, aber wenige Trades für Optimizer |
| 6 | **XRP 4h / 6h** | 60–70% | Nur in Bullphasen sinnvoll |
| 6 | **SUI 4h** | 58–68% | Gute Trends, aber wenig historische Daten |
| ❌ | **DOT / ADA / NEAR auf beliebigem TF** | < 60% | Zu viele Seitwärtsphasen für EMA-Bias |
| ❌ | **DOGE / SHIB auf 30m oder kürzer** | < 60% | Kein Fibonacci, kein Trend, hohe Noise-Rate |

> **Empfehlung für den Einstieg:** `BTC 4h` und `ETH 4h` gleichzeitig optimieren. Diese Kombination liefert zuverlässig valide Configs und ist das Herzstück jedes JaegerBot-Portfolios.

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
