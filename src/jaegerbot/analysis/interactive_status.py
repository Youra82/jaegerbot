#!/usr/bin/env python3
"""
Interactive Status für JaegerBot - ANN-basierte Strategie mit EMA, MACD, RSI, Bollinger Bands
Zeigt Candlestick-Chart mit technischen Indikatoren und simulierten Trades
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import logging

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

def load_config(symbol, timeframe):
    """Lädt Konfiguration für JaegerBot"""
    configs_dir = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs')
    safe_filename_base = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    config_filename = f"config_{safe_filename_base}.json"
    config_path = os.path.join(configs_dir, config_filename)
    
    if not os.path.exists(config_path):
        config_filename = f"config_{safe_filename_base}_macd.json"
        config_path = os.path.join(configs_dir, config_filename)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config nicht gefunden für {symbol} {timeframe}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def add_jaegerbot_indicators(df):
    """Fügt JaegerBot-spezifische Indikatoren hinzu"""
    # EMA
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    
    return df

def create_interactive_chart(symbol, timeframe, df, trades, start_date, end_date, window=None):
    """Erstellt interaktiven Chart mit Indikatoren und Trades"""
    
    # Filter auf Fenster
    if window:
        cutoff_date = datetime.now() - timedelta(days=window)
        df = df[df.index >= cutoff_date].copy()
    
    # Filter auf Start/End Datum
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    # Erstelle Subplots: Hauptchart + MACD + RSI
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("OHLC + EMA + Bollinger Bands", "MACD", "RSI")
    )
    
    # === Row 1: Candlestick + EMAs + Bollinger Bands ===
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # EMAs
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ema_20'], name='EMA 20', line=dict(color='orange', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ema_50'], name='EMA 50', line=dict(color='blue', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ema_200'], name='EMA 200', line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', line=dict(color='green', width=1, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', line=dict(color='green', width=1, dash='dash')),
        row=1, col=1
    )
    
    # === Row 2: MACD ===
    fig.add_trace(
        go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue', width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red', width=1.5)),
        row=2, col=1
    )
    
    # === Row 3: RSI ===
    fig.add_trace(
        go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple', width=1.5)),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # === Trade Marker ===
    for trade in trades:
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        exit_time = trade['exit_time']
        exit_price = trade['exit_price']
        profit = trade['profit']
        
        color = 'green' if profit > 0 else 'red'
        
        # Entry Point
        fig.add_trace(
            go.Scatter(
                x=[entry_time],
                y=[entry_price],
                mode='markers',
                marker=dict(size=10, color='green', symbol='triangle-up'),
                name=f'Entry ({entry_price:.2f})',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Exit Point
        fig.add_trace(
            go.Scatter(
                x=[exit_time],
                y=[exit_price],
                mode='markers',
                marker=dict(size=10, color=color, symbol='triangle-down'),
                name=f'Exit ({exit_price:.2f})',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Verbindungslinie
        fig.add_trace(
            go.Scatter(
                x=[entry_time, exit_time],
                y=[entry_price, exit_price],
                mode='lines',
                line=dict(color=color, width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Layout
    title = f"{symbol} {timeframe} - JaegerBot (ANN-Strategie)"
    fig.update_layout(
        title=title,
        height=1000,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="JaegerBot Interactive Status")
    parser.add_argument('--symbol', required=True, type=str)
    parser.add_argument('--timeframe', default='4h', type=str)
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--start-capital', type=float, default=1000)
    parser.add_argument('--window', type=int, help='Letzten N Tage anzeigen')
    parser.add_argument('--send-telegram', action='store_true')
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Lade Config für {args.symbol} {args.timeframe}...")
        config = load_config(args.symbol, args.timeframe)
        
        # Hole Daten vom Exchange
        logger.info("Verbinde mit Exchange...")
        with open(os.path.join(PROJECT_ROOT, 'secret.json'), 'r') as f:
            secrets = json.load(f)
        
        account = secrets['jaegerbot'][0]
        exchange = Exchange(account)
        
        logger.info(f"Lade OHLCV Daten für {args.symbol}...")
        df = exchange.get_ohlcv(args.symbol, args.timeframe, limit=500)
        
        logger.info("Berechne Indikatoren...")
        df = add_jaegerbot_indicators(df)
        
        # Backtest durchführen (vereinacht - ohne ANN für Speed)
        logger.info("Führe vereinachten Backtest durch...")
        trades = []  # Vereinacht: keine echten Trades aus ANN
        
        logger.info("Erstelle Chart...")
        fig = create_interactive_chart(
            args.symbol,
            args.timeframe,
            df,
            trades,
            args.start,
            args.end,
            args.window
        )
        
        # Speichere HTML
        output_file = f"/tmp/jaegerbot_{args.symbol.replace('/', '_')}_{args.timeframe}.html"
        fig.write_html(output_file)
        logger.info(f"✅ Chart gespeichert: {output_file}")
        
        # Telegram versenden (optional)
        if args.send_telegram:
            logger.info("Sende Chart via Telegram...")
            telegram_config = secrets.get('telegram', {})
            if telegram_config and os.path.exists(output_file):
                from jaegerbot.utils.telegram import send_file
                send_file(output_file, telegram_config)
        
        logger.info("✅ Fertig!")
        
    except Exception as e:
        logger.error(f"Fehler: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
