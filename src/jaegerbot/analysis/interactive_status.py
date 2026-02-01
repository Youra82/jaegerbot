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
    
    print("\n" + "="*60)
    print("Verfügbare Konfigurationen:")
    print("="*60)
    for idx, (filename, _) in enumerate(configs, 1):
        # Extrahiere Symbol/Timeframe aus Dateiname
        clean_name = filename.replace('config_', '').replace('.json', '')
        print(f"{idx:2d}) {clean_name}")
    print("="*60)
    
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
    # Kerzen-Daten sind bereits vorhanden, keine zusätzlichen Indikatoren nötig
    # Die eigentliche ANN-Analyse passiert in der Backtest-Funktion
    return df

def create_interactive_chart(symbol, timeframe, df, trades, equity_df, stats, start_date, end_date, window=None, start_capital=1000):
    """Erstellt interaktiven Chart mit Candlesticks, Trade-Signalen und Equity Curve"""
    from plotly.subplots import make_subplots
    
    # Filter auf Fenster
    if window:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=window)
        df = df[df.index >= cutoff_date].copy()
    
    # Filter auf Start/End Datum
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date, utc=True)]
    
    # Erstelle Chart mit secondary_y für Equity Curve
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # === Candlestick Chart (primäre Y-Achse) ===
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            showlegend=True
        ),
        secondary_y=False
    )
    
    # === Equity Curve (sekundäre Y-Achse, rechts) ===
    if not equity_df.empty and 'equity' in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                name='Kontostand',
                line=dict(color='#3b82f6', width=2),
                hovertemplate='<b>Kontostand</b><br>Zeit: %{x}<br>Equity: $%{y:.2f}<extra></extra>',
                showlegend=True
            ),
            secondary_y=True
        )
    
    # === Trade-Signale extrahieren und eintragen ===
    # Gruppiere Trades: Long (entry_long, exit_long) und Short (entry_short, exit_short)
    entry_long_x, entry_long_y = [], []
    exit_long_x, exit_long_y = [], []
    entry_short_x, entry_short_y = [], []
    exit_short_x, exit_short_y = [], []
    
    for trade in trades:
        # Entry Long (Dreieck nach oben, grün)
        if 'entry_long' in trade:
            entry_time = trade['entry_long'].get('time')
            entry_price = trade['entry_long'].get('price')
            if entry_time and entry_price:
                entry_long_x.append(pd.to_datetime(entry_time))
                entry_long_y.append(entry_price)
        
        # Exit Long (Kreis, Cyan)
        if 'exit_long' in trade:
            exit_time = trade['exit_long'].get('time')
            exit_price = trade['exit_long'].get('price')
            if exit_time and exit_price:
                exit_long_x.append(pd.to_datetime(exit_time))
                exit_long_y.append(exit_price)
        
        # Entry Short (Dreieck nach unten, Orange)
        if 'entry_short' in trade:
            entry_time = trade['entry_short'].get('time')
            entry_price = trade['entry_short'].get('price')
            if entry_time and entry_price:
                entry_short_x.append(pd.to_datetime(entry_time))
                entry_short_y.append(entry_price)
        
        # Exit Short (Diamant, Rot)
        if 'exit_short' in trade:
            exit_time = trade['exit_short'].get('time')
            exit_price = trade['exit_short'].get('price')
            if exit_time and exit_price:
                exit_short_x.append(pd.to_datetime(exit_time))
                exit_short_y.append(exit_price)
    
    # Entry Long: Dreieck nach oben, grün (#16a34a)
    if entry_long_x:
        fig.add_trace(go.Scatter(
            x=entry_long_x, y=entry_long_y, mode="markers",
            marker=dict(color="#16a34a", symbol="triangle-up", size=14, line=dict(width=1.2, color="#0f5132")),
            name="Entry Long",
            showlegend=True
        ), secondary_y=False)
    
    # Exit Long: Kreis, Cyan (#22d3ee)
    if exit_long_x:
        fig.add_trace(go.Scatter(
            x=exit_long_x, y=exit_long_y, mode="markers",
            marker=dict(color="#22d3ee", symbol="circle", size=12, line=dict(width=1.1, color="#0e7490")),
            name="Exit Long",
            showlegend=True
        ), secondary_y=False)
    
    # Entry Short: Dreieck nach unten, Orange (#f59e0b)
    if entry_short_x:
        fig.add_trace(go.Scatter(
            x=entry_short_x, y=entry_short_y, mode="markers",
            marker=dict(color="#f59e0b", symbol="triangle-down", size=14, line=dict(width=1.2, color="#92400e")),
            name="Entry Short",
            showlegend=True
        ), secondary_y=False)
    
    # Exit Short: Diamant, Rot (#ef4444)
    if exit_short_x:
        fig.add_trace(go.Scatter(
            x=exit_short_x, y=exit_short_y, mode="markers",
            marker=dict(color="#ef4444", symbol="diamond", size=12, line=dict(width=1.1, color="#7f1d1d")),
            name="Exit Short",
            showlegend=True
        ), secondary_y=False)
    
    # === Stats für Titel extrahieren ===
    end_capital = stats.get('end_capital', start_capital)
    pnl_pct = ((end_capital - start_capital) / start_capital) * 100 if start_capital > 0 else 0
    max_dd = stats.get('max_drawdown', 0)
    total_trades = stats.get('total_trades', len(trades))
    win_rate = stats.get('win_rate', 0)
    
    # Layout mit Stats im Titel
    title = (f"{symbol} {timeframe} - JaegerBot (ANN) | "
             f"Kapital: ${start_capital:.0f}→${end_capital:.0f} | "
             f"PnL: {pnl_pct:+.2f}% | DD: {max_dd:.1f}% | "
             f"Trades: {total_trades} | WR: {win_rate:.1f}%")
    
    fig.update_layout(
        title=title,
        height=650,
        hovermode='x unified',
        template='plotly_white',
        dragmode='zoom',
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.05), fixedrange=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True
    )
    
    # Primäre Y-Achse (links): Preis
    fig.update_yaxes(title_text="Preis", secondary_y=False, fixedrange=False)
    
    # Sekundäre Y-Achse (rechts): Kontostand
    fig.update_yaxes(title_text="Kontostand ($)", secondary_y=True, fixedrange=False, showgrid=False)
    
    # X-Achse zoomen erlauben
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
    
    # Generiere Chart für jede gewählte Config
    for filename, filepath in selected_configs:
        try:
            logger.info(f"\nVerarbeite {filename}...")
            
            config = load_config(filepath)
            symbol = config['market']['symbol']
            timeframe = config['market']['timeframe']
            
            logger.info(f"Lade OHLCV-Daten für {symbol} {timeframe}...")
            
            # Nutze historische Daten basierend auf Start/End Datum
            # Falls keine Daten angefordert: letzte 30 Tage
            if not start_date:
                start_date_for_load = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
            else:
                start_date_for_load = start_date
            
            if not end_date:
                end_date_for_load = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            else:
                end_date_for_load = end_date
            
            df = exchange.fetch_historical_ohlcv(symbol, timeframe, start_date_for_load, end_date_for_load)
            
            if df is None or len(df) == 0:
                logger.warning(f"Keine Daten für {symbol} {timeframe} im Zeitraum {start_date_for_load} bis {end_date_for_load}")
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
            
            backtest_result, equity_snapshots = run_ann_backtest(
                df, 
                config,
                model_paths,
                start_capital=1000,
                use_macd_filter=config.get('market', {}).get('use_macd_filter', False),
                timeframe=timeframe,
                verbose=False,
                return_equity=True
            )
            
            # Extrahiere Trades und Stats aus Backtest-Ergebnis
            trades = backtest_result.get('trades', [])
            
            # Equity Snapshots in DataFrame umwandeln
            if equity_snapshots:
                equity_df = pd.DataFrame(equity_snapshots)
                equity_df.set_index('timestamp', inplace=True)
            else:
                equity_df = pd.DataFrame()
            
            # Erstelle Chart mit Trades und Equity Curve
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
                1000
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
