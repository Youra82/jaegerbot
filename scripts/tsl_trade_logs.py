#!/usr/bin/env python3
import os
import json
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.analysis.backtester import load_data, calculate_supertrend_direction
from jaegerbot.utils.ann_model import load_model_and_scaler, create_ann_features

import csv

config_path = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs', 'config_SOLUSDTUSDT_5m.json')
with open(config_path, 'r') as f:
    cfg = json.load(f)

symbol = cfg['market']['symbol']
timeframe = cfg['market']['timeframe']

sym_key = symbol.replace('/', '').replace(':', '')
model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{sym_key}_{timeframe}.h5')
scaler_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{sym_key}_{timeframe}.joblib')

model, scaler = load_model_and_scaler(model_path, scaler_path)
if model is None:
    raise SystemExit('Model not found')

start_date = '2025-10-01'
end_date = '2025-12-17'
raw = load_data(symbol, timeframe, start_date, end_date)
if raw.empty:
    raise SystemExit('No data loaded')

data = create_ann_features(raw.copy())
data.dropna(inplace=True)

# compute supertrend direction
st = calculate_supertrend_direction(data)

# merge
data['supertrend_direction'] = st
data.dropna(inplace=True)

feature_cols = [
    'bb_width', 'bb_pband', 'obv', 'rsi', 'macd_diff', 'macd', 
    'atr_normalized', 'adx', 'adx_pos', 'adx_neg',
    'volume_ratio', 'mfi', 'cmf',
    'price_to_ema20', 'price_to_ema50',
    'stoch_k', 'stoch_d', 'williams_r', 'roc', 'cci',
    'price_to_resistance', 'price_to_support',
    'high_low_range', 'close_to_high', 'close_to_low',
    'day_of_week', 'hour_of_day',
    'returns_lag1', 'returns_lag2', 'returns_lag3', 'hist_volatility'
]

X = data[feature_cols]
features_scaled = scaler.transform(X)
preds = model.predict(features_scaled, verbose=0).flatten()

data['prediction'] = preds

activation_values = [1.5, 1.0]

for activation in activation_values:
    rows = []
    activation_rr = activation
    callback_rate = cfg['risk'].get('trailing_stop_callback_rate_pct', 1.0) / 100.0
    initial_sl_pct = cfg['risk'].get('min_sl_pct', 1.0) / 100.0
    leverage = cfg['risk'].get('leverage', 10)
    risk_per_trade = cfg['risk'].get('risk_per_trade_pct', 1.0) / 100.0
    risk_reward = cfg['risk'].get('risk_reward_ratio', 2.0)

    position = None
    trade_id = 0

    for idx in range(len(data)):
        row = data.iloc[idx]
        ts = row.name
        if position:
            exit_price = None
            exit_reason = None
            # long
            if position['side'] == 'long':
                if not position['trailing_active'] and row['high'] >= position['activation_price']:
                    position['trailing_active'] = True
                    position['activation_index'] = idx
                    position['activation_time'] = ts
                if position['trailing_active']:
                    position['peak_price'] = max(position['peak_price'], row['high'])
                    trailing_sl = position['peak_price'] * (1 - position['callback_rate'])
                    position['stop_loss'] = max(position['stop_loss'], trailing_sl)
                if row['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'TSL'
                elif not position['trailing_active'] and row['high'] >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'TP'
            else:
                # short
                if not position['trailing_active'] and row['low'] <= position['activation_price']:
                    position['trailing_active'] = True
                    position['activation_index'] = idx
                    position['activation_time'] = ts
                if position['trailing_active']:
                    position['peak_price'] = min(position['peak_price'], row['low'])
                    trailing_sl = position['peak_price'] * (1 + position['callback_rate'])
                    position['stop_loss'] = min(position['stop_loss'], trailing_sl)
                if row['high'] >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'TSL'
                elif not position['trailing_active'] and row['low'] <= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'TP'

            if exit_price is not None:
                pnl_pct = (exit_price / position['entry_price'] - 1) if position['side'] == 'long' else (1 - exit_price / position['entry_price'])
                notional = position['notional']
                pnl = notional * pnl_pct
                fees = notional * 0.0005 * 2
                net = pnl - fees
                # cap
                risk_amount = 1000.0 * position['risk_per_trade']
                if net < -risk_amount: net = -risk_amount
                max_profit = risk_amount * position['risk_reward']
                if net > max_profit: net = max_profit

                rows.append({
                    'trade_id': position['trade_id'],
                    'activation_rr': activation_rr,
                    'side': position['side'],
                    'entry_time': position['entry_time'].isoformat(),
                    'entry_price': position['entry_price'],
                    'activation_time': position.get('activation_time').isoformat() if position.get('activation_time') is not None else None,
                    'activation_price': position.get('activation_price'),
                    'peak_price': position.get('peak_price'),
                    'exit_time': ts.isoformat(),
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': net
                })

                position = None

        # open
        if not position:
            side = 'long' if row['prediction'] >= cfg.get('strategy', {}).get('prediction_threshold', 0.6) else 'short' if row['prediction'] <= (1 - cfg.get('strategy', {}).get('prediction_threshold', 0.6)) else None
            if side:
                st = row.get('supertrend_direction', None)
                if st == 1.0 and side == 'short':
                    continue
                if st == -1.0 and side == 'long':
                    continue
                entry_price = row['close']
                sl_distance = entry_price * initial_sl_pct
                if sl_distance <= 0: continue
                notional = (1000.0 * risk_per_trade) / initial_sl_pct
                margin_used = notional / leverage
                if margin_used > 1000.0: continue
                take_profit = entry_price + sl_distance * risk_reward if side == 'long' else entry_price - sl_distance * risk_reward
                activation_price = entry_price + sl_distance * activation_rr if side == 'long' else entry_price - sl_distance * activation_rr
                trade_id += 1
                position = {'trade_id': trade_id, 'side': side, 'entry_time': ts, 'entry_price': entry_price,
                            'stop_loss': entry_price - sl_distance if side=='long' else entry_price + sl_distance,
                            'take_profit': take_profit, 'notional': notional, 'margin_used': margin_used,
                            'trailing_active': False, 'activation_price': activation_price, 'peak_price': entry_price,
                            'callback_rate': callback_rate, 'risk_per_trade': risk_per_trade, 'risk_reward': risk_reward,
                            'activation_index': None}

    out_dir = os.path.join(PROJECT_ROOT, 'scripts', 'tsl_logs')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'tsl_logs_{symbol.replace("/","-").replace(":","-")}_{timeframe}_act{activation_rr}.csv')

    with open(out_file, 'w', newline='') as csvfile:
        fieldnames = ['trade_id','activation_rr','side','entry_time','entry_price','activation_time','activation_price','peak_price','exit_time','exit_price','exit_reason','pnl']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote {len(rows)} trades to {out_file}')
    # print first 5 rows
    for r in rows[:5]:
        print(r)

print('Done')
