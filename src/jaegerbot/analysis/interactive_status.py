#!/usr/bin/env python3
"""
Interactive Charts für JaegerBot - ANN-basierte Strategie
Zeigt Candlestick-Chart mit Trade-Signalen (Entry/Exit Long/Short)
Nutzt durchnummerierte Konfigurationsdateien zum Auswählen
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.exchange import Exchange
from jaegerbot.analysis.backtester import run_ann_backtest

def setup_logging():
    logger = logging.getLogger('interactive_status')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(ch)
    return logger

logger = setup_logging()

def get_config_files():
    """Sucht alle Konfigurationsdateien auf"""
    configs_dir = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs')
    if not os.path.exists(configs_dir):
        return []
    
    configs = []
    for filename in sorted(os.listdir(configs_dir)):
        if filename.startswith('config_') and filename.endswith('.json'):
            filepath = os.path.join(configs_dir, filename)
            configs.append((filename, filepath))
    
    return configs

def select_configs():
    """Zeigt durchnummerierte Konfigurationsdateien und lässt User wählen"""
    configs = get_config_files()

    if not configs:
        logger.error("Keine Konfigurationsdateien gefunden!")
        sys.exit(1)

    print("\n" + "="*70)
    print("Verfügbare Konfigurationen:")
    print("="*70)
    for idx, (filename, filepath) in enumerate(configs, 1):
        clean_name = filename.replace('config_', '').replace('.json', '')
        # Lade _meta.pnl_pct aus Config-Datei
        pnl_str = ""
        try:
            with open(filepath, 'r') as f:
                cfg = json.load(f)
            pnl_pct = cfg.get('_meta', {}).get('pnl_pct')
            if pnl_pct is not None:
                sign = "+" if pnl_pct >= 0 else ""
                pnl_str = f"  [{sign}{pnl_pct:.1f}%]"
        except Exception:
            pass
        print(f"{idx:2d}) {clean_name}{pnl_str}")
    print("="*70)
    
    print("\nWähle Konfiguration(en) zum Anzeigen:")
    print("  Einzeln: z.B. '1' oder '5'")
    print("  Mehrfach: z.B. '1,3,5' oder '1 3 5'")
    
    selection = input("\nAuswahl: ").strip()
    
    # Parse Eingabe
    selected_indices = []
    for part in selection.replace(',', ' ').split():
        try:
            idx = int(part)
            if 1 <= idx <= len(configs):
                selected_indices.append(idx - 1)
            else:
                logger.warning(f"Index {idx} außerhalb des Bereichs")
        except ValueError:
            logger.warning(f"Ungültige Eingabe: {part}")
    
    if not selected_indices:
        logger.error("Keine gültigen Konfigurationen gewählt!")
        sys.exit(1)
    
    return [configs[i] for i in selected_indices]

def load_config(filepath):
    """Lädt eine Konfiguration"""
    with open(filepath, 'r') as f:
        return json.load(f)

def add_jaegerbot_indicators(df):
    """Fügt Indikatoren für Chart-Anzeige hinzu (vereinfacht)"""
    return df


def _params_from_config(config: dict) -> dict:
    """Flacht die verschachtelte Config-JSON zu flachen Backtest-Params."""
    risk = config.get('risk', {})
    strategy = config.get('strategy', {})
    return {
        'prediction_threshold':         strategy.get('prediction_threshold', 0.6),
        'risk_reward_ratio':            risk.get('risk_reward_ratio', 1.5),
        'risk_per_trade_pct':           risk.get('risk_per_trade_pct', 1.0),
        'leverage':                     risk.get('leverage', 10),
        'initial_sl_pct':               risk.get('min_sl_pct', 1.0),
        'atr_multiplier_sl':            risk.get('atr_multiplier_sl', 2.0),
        'min_sl_pct':                   risk.get('min_sl_pct', 1.0),
        'trailing_stop_activation_rr':  risk.get('trailing_stop_activation_rr', 2.0),
        'trailing_stop_callback_rate_pct': risk.get('trailing_stop_callback_rate_pct', 1.0),
    }


def build_equity_curve(equity_snapshots: list, start_capital: float) -> pd.DataFrame:
    """Baut Equity-DataFrame aus den Snapshots des Backtesters."""
    if not equity_snapshots:
        return pd.DataFrame()
    equity_df = pd.DataFrame(equity_snapshots)
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'], utc=True)
    equity_df.set_index('timestamp', inplace=True)
    # Startpunkt voranstellen
    start_row = pd.DataFrame([{'equity': start_capital}],
                              index=pd.DatetimeIndex([equity_df.index[0] - pd.Timedelta(seconds=1)], tz='UTC'))
    return pd.concat([start_row, equity_df])


def create_interactive_chart(symbol, timeframe, df, trades, equity_df, stats,
                              start_date, end_date, window=None, start_capital=1000):
    """Erstellt interaktiven Chart mit separatem Equity-Panel (oben) und Candlestick (unten)."""

    # Filter df auf Anzeige-Zeitraum
    if window:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=window)
        df = df[df.index >= cutoff_date].copy()
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date, utc=True)]

    # Zeitgrenzen für Trades-Filter
    t_min = df.index[0] if not df.empty else None
    t_max = df.index[-1] if not df.empty else None

    # === 2 Subplots: Equity oben (30%), Candlestick unten (70%) ===
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Kontostand ($)", f"{symbol} {timeframe}"),
    )

    # === Equity-Kurve (oben, Row 1) ===
    if not equity_df.empty and 'equity' in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index, y=equity_df['equity'],
                name='Kontostand',
                line=dict(color='#2563eb', width=2),
                fill='tozeroy',
                fillcolor='rgba(37,99,235,0.10)',
                hovertemplate='<b>Kontostand</b>: $%{y:.2f}<extra></extra>',
                showlegend=True,
            ),
            row=1, col=1,
        )
    else:
        # Keine Trades → flache Startkapital-Linie
        if not df.empty:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[0], df.index[-1]],
                    y=[start_capital, start_capital],
                    name='Kontostand',
                    line=dict(color='#94a3b8', width=1.5, dash='dot'),
                    hovertemplate='Kontostand: $%{y:.2f}<extra></extra>',
                    showlegend=True,
                ),
                row=1, col=1,
            )

    # === Candlestick (unten, Row 2) ===
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='OHLC',
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            showlegend=True,
        ),
        row=2, col=1,
    )

    # === Trade-Signale (Row 2, nur innerhalb Anzeige-Zeitraum) ===
    entry_long_x, entry_long_y   = [], []
    exit_long_x,  exit_long_y    = [], []
    entry_short_x, entry_short_y = [], []
    exit_short_x,  exit_short_y  = [], []

    for trade in trades:
        if 'entry_long' in trade:
            t = trade['entry_long']
            ts = pd.to_datetime(t['time'])
            if t_min is None or (ts >= t_min and ts <= t_max):
                entry_long_x.append(ts); entry_long_y.append(t['price'])
        if 'exit_long' in trade:
            t = trade['exit_long']
            ts = pd.to_datetime(t['time'])
            if t_min is None or (ts >= t_min and ts <= t_max):
                exit_long_x.append(ts); exit_long_y.append(t['price'])
        if 'entry_short' in trade:
            t = trade['entry_short']
            ts = pd.to_datetime(t['time'])
            if t_min is None or (ts >= t_min and ts <= t_max):
                entry_short_x.append(ts); entry_short_y.append(t['price'])
        if 'exit_short' in trade:
            t = trade['exit_short']
            ts = pd.to_datetime(t['time'])
            if t_min is None or (ts >= t_min and ts <= t_max):
                exit_short_x.append(ts); exit_short_y.append(t['price'])

    if entry_long_x:
        fig.add_trace(go.Scatter(x=entry_long_x, y=entry_long_y, mode="markers",
            marker=dict(color="#16a34a", symbol="triangle-up", size=14, line=dict(width=1.2, color="#0f5132")),
            name="Entry Long", showlegend=True), row=2, col=1)
    if exit_long_x:
        fig.add_trace(go.Scatter(x=exit_long_x, y=exit_long_y, mode="markers",
            marker=dict(color="#22d3ee", symbol="circle", size=12, line=dict(width=1.1, color="#0e7490")),
            name="Exit Long", showlegend=True), row=2, col=1)
    if entry_short_x:
        fig.add_trace(go.Scatter(x=entry_short_x, y=entry_short_y, mode="markers",
            marker=dict(color="#f59e0b", symbol="triangle-down", size=14, line=dict(width=1.2, color="#92400e")),
            name="Entry Short", showlegend=True), row=2, col=1)
    if exit_short_x:
        fig.add_trace(go.Scatter(x=exit_short_x, y=exit_short_y, mode="markers",
            marker=dict(color="#ef4444", symbol="diamond", size=12, line=dict(width=1.1, color="#7f1d1d")),
            name="Exit Short", showlegend=True), row=2, col=1)

    # === Titel mit Stats ===
    end_capital  = stats.get('end_capital', start_capital)
    pnl_pct      = stats.get('total_pnl_pct', 0)
    total_trades = stats.get('trades_count', 0)
    win_rate     = stats.get('win_rate', 0)
    raw_dd       = stats.get('max_drawdown_pct', 0)
    max_dd       = raw_dd * 100 if total_trades > 0 else 0.0
    title = (f"{symbol} {timeframe} - JaegerBot (ANN) | "
             f"${start_capital:.0f}→${end_capital:.0f} | "
             f"PnL: {pnl_pct:+.2f}% | DD: {max_dd:.1f}% | "
             f"Trades: {total_trades} | WR: {win_rate:.1f}%")

    # Annotation wenn keine Trades (im unteren Panel)
    annotations = []
    if total_trades == 0:
        annotations.append(dict(
            text="⚠ Keine Signale im gewählten Zeitraum",
            xref="paper", yref="y2",
            x=0.5, y=(df['close'].mean() if not df.empty else 0),
            showarrow=False,
            font=dict(size=16, color="#94a3b8"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#94a3b8",
            borderwidth=1,
        ))

    fig.update_layout(
        title=title,
        height=750,
        hovermode='x unified',
        template='plotly_white',
        dragmode='zoom',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
        annotations=annotations,
        xaxis2=dict(rangeslider=dict(visible=True, thickness=0.04)),
    )
    fig.update_yaxes(title_text="Kontostand ($)", row=1, col=1, fixedrange=False)
    fig.update_yaxes(title_text="Preis", row=2, col=1, fixedrange=False)
    fig.update_xaxes(fixedrange=False)

    return fig

def main():
    # Wähle Konfigurationsdateien
    selected_configs = select_configs()
    
    # Parameter für Chart-Generierung
    print("\n" + "="*60)
    print("Chart-Optionen:")
    print("="*60)
    
    start_date = input("Startdatum (YYYY-MM-DD) [leer=beliebig]: ").strip() or None
    end_date = input("Enddatum (YYYY-MM-DD) [leer=heute]: ").strip() or None
    window_input = input("Letzten N Tage anzeigen [leer=alle]: ").strip()
    window = int(window_input) if window_input.isdigit() else None
    capital_input = input("Startkapital in $ [Standard: 1000]: ").strip()
    try:
        start_capital = float(capital_input) if capital_input else 1000.0
    except ValueError:
        logger.warning(f"Ungültiges Startkapital '{capital_input}', verwende $1000.")
        start_capital = 1000.0
    send_telegram = input("Telegram versenden? (j/n) [Standard: n]: ").strip().lower() in ['j', 'y', 'yes']
    
    try:
        with open(os.path.join(PROJECT_ROOT, 'secret.json'), 'r') as f:
            secrets = json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Laden von secret.json: {e}")
        sys.exit(1)
    
    account = secrets.get('jaegerbot', [None])[0]
    if not account:
        logger.error("Keine Jaegerbot-Accountkonfiguration gefunden")
        sys.exit(1)
    
    exchange = Exchange(account)
    telegram_config = secrets.get('telegram', {})
    
    # Warmup-Tage je Timeframe (damit Indikatoren/Features genug History haben)
    WARMUP_DAYS = {
        '1m': 3, '3m': 5, '5m': 7, '15m': 10, '30m': 14,
        '1h': 20, '2h': 25, '4h': 30, '6h': 40, '8h': 45,
        '12h': 60, '1d': 90,
    }

    # Generiere Chart für jede gewählte Config
    for filename, filepath in selected_configs:
        try:
            logger.info(f"\nVerarbeite {filename}...")

            config = load_config(filepath)
            symbol = config['market']['symbol']
            timeframe = config['market']['timeframe']

            logger.info(f"Lade OHLCV-Daten für {symbol} {timeframe}...")

            # Berechne Lade-Startdatum (inkl. Warmup-Puffer für Feature-Engineering)
            warmup = WARMUP_DAYS.get(timeframe, 30)

            if not start_date:
                display_start = datetime.now(timezone.utc) - timedelta(days=30)
            else:
                display_start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            # Warmup-Datum: genug History vor dem eigentlichen Startdatum
            load_start = (display_start - timedelta(days=warmup)).strftime("%Y-%m-%d")

            if not end_date:
                end_date_for_load = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            else:
                end_date_for_load = end_date

            df = exchange.fetch_historical_ohlcv(symbol, timeframe, load_start, end_date_for_load)
            
            if df is None or len(df) == 0:
                logger.warning(f"Keine Daten für {symbol} {timeframe} im Zeitraum {load_start} bis {end_date_for_load}")
                continue
            
            logger.info("Verarbeite Daten...")
            df = add_jaegerbot_indicators(df)
            
            # Führe Backtest durch, um Trades zu generieren
            logger.info("Führe Backtest durch...")
            from jaegerbot.analysis.backtester import run_ann_backtest
            
            model_save_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', 
                                          f'ann_predictor_{symbol.replace("/", "").replace(":", "")}_{timeframe}.h5')
            scaler_save_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', 
                                           f'ann_scaler_{symbol.replace("/", "").replace(":", "")}_{timeframe}.joblib')
            
            model_paths = {'model': model_save_path, 'scaler': scaler_save_path}

            # Config-JSON in flache Backtest-Parameter umwandeln
            params = _params_from_config(config)

            backtest_result = run_ann_backtest(
                df,
                params,
                model_paths,
                start_capital=start_capital,
                use_macd_filter=config.get('behavior', {}).get('use_macd_filter', False),
                timeframe=timeframe,
                verbose=False,
            )

            # Trades + Equity aus Backtest-Ergebnis
            trades           = backtest_result.get('trades', [])
            equity_snapshots = backtest_result.get('equity_snapshots', [])
            equity_df        = build_equity_curve(equity_snapshots, start_capital=start_capital)

            # Erstelle Chart
            logger.info("Erstelle Chart...")
            fig = create_interactive_chart(
                symbol,
                timeframe,
                df,
                trades,
                equity_df,
                backtest_result,
                start_date,
                end_date,
                window,
                start_capital=start_capital,
            )
            
            # Speichere HTML
            safe_name = f"{symbol.replace('/', '_')}_{timeframe}"
            output_file = f"/tmp/jaegerbot_{safe_name}.html"
            fig.write_html(output_file)
            logger.info(f"✅ Chart gespeichert: {output_file}")
            
            # Telegram versenden (optional)
            if send_telegram and telegram_config:
                try:
                    logger.info(f"Sende Chart via Telegram...")
                    from jaegerbot.utils.telegram import send_document
                    bot_token = telegram_config.get('bot_token')
                    chat_id = telegram_config.get('chat_id')
                    if bot_token and chat_id:
                        send_document(bot_token, chat_id, output_file, caption=f"Chart: {symbol} {timeframe}")
                except Exception as e:
                    logger.warning(f"Konnte Chart nicht via Telegram versenden: {e}")
        
        except Exception as e:
            logger.error(f"Fehler bei {filename}: {e}", exc_info=True)
            continue
    
    logger.info("\n✅ Alle Charts generiert!")

if __name__ == '__main__':
    main()
