# src/jaegerbot/utils/ann_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import logging
import ta
import os

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def create_ann_features(df):
    # Bestehende Features
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_width'] = bollinger.bollinger_wband()
    df['bb_pband'] = bollinger.bollinger_pband()  # Position innerhalb der Bänder
    df['bb_hband'] = bollinger.bollinger_hband()
    df['bb_lband'] = bollinger.bollinger_lband()
    
    if 'volume' in df.columns and df['volume'].sum() > 0:
        df['obv'] = ta.volume.on_balance_volume(close=df['close'], volume=df['volume'])
        # NEU: Volume-Features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)
        df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=14)
    else:
        df['obv'] = 0
        df['volume_sma'] = 0
        df['volume_ratio'] = 1
        df['mfi'] = 50
        df['cmf'] = 0
        df['vwap'] = df['close']
    
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD Features
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd_diff'] = macd.macd_diff()
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # ATR
    atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr_indicator.average_true_range()
    df['atr_normalized'] = (df['atr'] / df['close']) * 100

    # *** ADX WIEDER HINZUGEFÜGT (KRITISCH!) ***
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
    df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)

    # NEU: EMA Trend-Features
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['price_to_ema20'] = (df['close'] - df['ema20']) / df['ema20']
    df['price_to_ema50'] = (df['close'] - df['ema50']) / df['ema50']
    
    # NEU: Momentum-Features
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
    df['roc'] = ta.momentum.roc(df['close'], window=12)
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    
    # NEU: Volatilität-Features
    df['keltner_channel_hband'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'], window=20)
    df['keltner_channel_lband'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'], window=20)
    df['donchian_channel_hband'] = ta.volatility.donchian_channel_hband(df['high'], df['low'], df['close'], window=20)
    df['donchian_channel_lband'] = ta.volatility.donchian_channel_lband(df['high'], df['low'], df['close'], window=20)
    
    # NEU: Support/Resistance Features
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()
    df['price_to_resistance'] = (df['resistance'] - df['close']) / df['close']
    df['price_to_support'] = (df['close'] - df['support']) / df['close']
    
    # NEU: Price Action Features
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 0.0001)
    df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
    
    # Zeitliche Features
    df['day_of_week'] = df.index.dayofweek
    df['hour_of_day'] = df.index.hour if hasattr(df.index, 'hour') else 0
    
    # Returns
    df['returns_lag1'] = df['close'].pct_change().shift(1)
    df['returns_lag2'] = df['close'].pct_change().shift(2)
    df['returns_lag3'] = df['close'].pct_change().shift(3)
    
    # NEU: Historical Volatility
    df['hist_volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    # ── CANDLE DNA FEATURES (dnabot-inspired) ──────────────────────────
    df['body_size']          = (df['close'] - df['open']).abs()
    df['candle_range']       = df['high'] - df['low']
    df['body_to_atr']        = df['body_size'] / (df['atr'] + 1e-9)
    df['upper_wick']         = df['high'] - df[['close','open']].max(axis=1)
    df['lower_wick']         = df[['close','open']].min(axis=1) - df['low']
    df['upper_wick_ratio']   = df['upper_wick'] / (df['candle_range'] + 1e-9)
    df['lower_wick_ratio']   = df['lower_wick'] / (df['candle_range'] + 1e-9)
    df['candle_direction']   = np.sign(df['close'] - df['open'])
    df['body_midpoint_ratio']= ((df[['close','open']].mean(axis=1) - df['low']) / (df['candle_range'] + 1e-9))
    df['bull_streak']        = (df['candle_direction'] == 1).rolling(5).sum()
    df['bear_streak']        = (df['candle_direction'] == -1).rolling(5).sum()

    # ── PIVOT / STRUCTURE FEATURES (stbot-inspired) ────────────────────
    _pw = 10
    df['pivot_high_live'] = df['high'].rolling(_pw * 2).max()
    df['pivot_low_live']  = df['low'].rolling(_pw * 2).min()
    df['dist_to_struct_high'] = (df['pivot_high_live'] - df['close']) / (df['atr'] + 1e-9)
    df['dist_to_struct_low']  = (df['close'] - df['pivot_low_live'])  / (df['atr'] + 1e-9)
    df['price_in_range_20']   = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-9)
    df['price_in_range_50']   = (df['close'] - df['low'].rolling(50).min()) / (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-9)

    # ── FIBONACCI ZONE FEATURE (vbot-inspired) ─────────────────────────
    _rh10 = df['high'].shift(1).rolling(10).max()
    _rl10 = df['low'].shift(1).rolling(10).min()
    _rr10 = _rh10 - _rl10
    df['fib_position'] = (df['close'] - _rl10) / (_rr10 + 1e-9)
    df['in_fib_zone']  = ((df['close'] >= _rl10 + _rr10 * 0.382) & (df['close'] <= _rl10 + _rr10 * 0.618)).astype(float)

    # ── VOLUME DIRECTION FEATURES ──────────────────────────────────────
    df['volume_direction']  = df['candle_direction'] * df['volume_ratio']
    df['buying_pressure']   = df['close_to_low']  * df['volume_ratio']
    df['selling_pressure']  = df['close_to_high'] * df['volume_ratio']

    return df

def prepare_data_for_ann(df, timeframe: str, verbose: bool = True):
    """
    Definiert ein "gutes Signal" basierend auf einem dynamischen Threshold,
    der sich an der Volatilität (ATR) des jeweiligen Datensatzes orientiert.
    """
    df_with_features = create_ann_features(df.copy())
    df_with_features.dropna(inplace=True)
    if df_with_features.empty:
        return pd.DataFrame(), pd.Series()

    # Adaptive 'lookahead' und 'volatility_multiplier' je nach Zeitfenster definieren
    if 'm' in timeframe:
        lookahead = 12
        volatility_multiplier = 2.5
    elif 'h' in timeframe:
        try:
            tf_num = int(timeframe.replace('h', ''))
            if tf_num == 1:
                lookahead = 8
                volatility_multiplier = 2.0
            elif tf_num <= 4:
                lookahead = 5
                volatility_multiplier = 1.75
            else: # 6h, 12h etc.
                lookahead = 4
                volatility_multiplier = 1.75
        except ValueError:
            lookahead = 5
            volatility_multiplier = 1.75
    elif 'd' in timeframe:
        lookahead = 5
        volatility_multiplier = 1.5
    else: # Fallback
        lookahead = 5
        volatility_multiplier = 2.0

    # ── TP/SL OUTCOME LABELS ───────────────────────────────────────────
    df_signals = df_with_features.copy()

    if 'm' in timeframe:
        _sl_mult, _rr, _max_fwd = 2.0, 2.0, 24
    elif timeframe in ('1h', '2h'):
        _sl_mult, _rr, _max_fwd = 2.0, 2.0, 16
    elif timeframe in ('4h', '6h'):
        _sl_mult, _rr, _max_fwd = 2.0, 2.0, 12
    else:
        _sl_mult, _rr, _max_fwd = 1.5, 2.0, 8

    _sl_arr  = df_signals['atr'].values * _sl_mult
    _tp_arr  = _sl_arr * _rr
    _closes  = df_signals['close'].values
    _highs   = df_signals['high'].values
    _lows    = df_signals['low'].values
    _n       = len(df_signals)
    _targets = np.zeros(_n, dtype=int)

    for _i in range(_n - _max_fwd):
        _tp_p = _closes[_i] + _tp_arr[_i]
        _sl_p = _closes[_i] - _sl_arr[_i]
        for _j in range(_i + 1, _i + 1 + _max_fwd):
            if _lows[_j] <= _sl_p:
                _targets[_i] = -1
                break
            if _highs[_j] >= _tp_p:
                _targets[_i] = 1
                break

    df_signals['target']        = _targets
    df_signals                  = df_signals[df_signals['target'] != 0].copy()
    df_signals['target_binary'] = (df_signals['target'] == 1).astype(int)
    df_with_features = df_signals

    # *** ERWEITERTE FEATURE-LISTE MIT ALLEN NEUEN FEATURES ***
    # Feature 'ema_cross_20_50' entfernt aufgrund durchgehend niedriger Wichtigkeit (<0.2%)
    feature_cols = [
        # Basis-Features
        'bb_width', 'bb_pband', 'obv', 'rsi', 'macd_diff', 'macd', 
        'atr_normalized', 'adx', 'adx_pos', 'adx_neg',
        
        # Volume-Features
        'volume_ratio', 'mfi', 'cmf',
        
        # Trend-Features
        'price_to_ema20', 'price_to_ema50',
        
        # Momentum-Features
        'stoch_k', 'stoch_d', 'williams_r', 'roc', 'cci',
        
        # Support/Resistance
        'price_to_resistance', 'price_to_support',
        
        # Price Action
        'high_low_range', 'close_to_high', 'close_to_low',
        
        # Zeitliche Features
        'day_of_week', 'hour_of_day',
        
        # Returns & Volatilität
        'returns_lag1', 'returns_lag2', 'returns_lag3', 'hist_volatility',

        # Candle DNA Features
        'body_to_atr', 'upper_wick_ratio', 'lower_wick_ratio',
        'candle_direction', 'body_midpoint_ratio',
        'bull_streak', 'bear_streak',

        # Pivot / Structure Features
        'dist_to_struct_high', 'dist_to_struct_low',
        'price_in_range_20', 'price_in_range_50',

        # Fibonacci Zone Features
        'fib_position', 'in_fib_zone',

        # Volume Direction Features
        'volume_direction', 'buying_pressure', 'selling_pressure',
    ]
    # ---

    X = df_with_features[feature_cols]
    y = df_with_features['target_binary']

    return X, y

def build_and_train_model(X_train, y_train):
    """
    Verbessertes Modell mit mehr Kapazität für die erweiterten Features.
    256-128-64-32 Architektur mit Batch Normalization für bessere Stabilität.
    """
    model = tf.keras.models.Sequential([
        # Layer 1: Mehr Neuronen für komplexere Feature-Interaktionen
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        # Layer 2
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.35),

        # Layer 3
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        # Layer 4
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        
        # Output
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Verwende Adam mit reduzierter Learning Rate für bessere Konvergenz
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks: Early Stopping und Learning Rate Reduction
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    model.fit(
        X_train, y_train, 
        validation_split=0.2, 
        epochs=150,  # Mehr Epochs mit Early Stopping
        batch_size=32, 
        callbacks=[early_stopping, reduce_lr], 
        verbose=1
    )
    
    return model

def save_model_and_scaler(model, scaler, model_path, scaler_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Modell & Scaler gespeichert.")

def load_model_and_scaler(model_path, scaler_path):
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        logging.info(f"Modell & Scaler geladen.")
        return model, scaler
    except Exception as e:
        logging.error(f"Fehler beim Laden von Modell/Scaler: {e}")
        return None, None
