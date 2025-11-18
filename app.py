from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_session import Session
import pandas as pd
import requests
import json
import os
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import threading
import logging
import time
import numpy as np
from dotenv import load_dotenv
import hashlib
import secrets

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'trading-signal-secret-2024')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
Session(app)

# Trading Configuration
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"
]

INTERVAL = "15m"
LIMIT = 300
SQUEEZE_THRESHOLD = 0.018
COOLDOWN_MINUTES = 60
SCAN_INTERVAL_MINUTES = 6
RISK_PER_TRADE = 0.01

# Admin credentials (hashed)
ADMIN_USERNAME = "BangNguyen89"
ADMIN_PASSWORD_HASH = hashlib.sha256("NLB@0708.".encode()).hexdigest()

# Thread safety
data_lock = threading.Lock()

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def load_json(file):
    """Load JSON file with error recovery"""
    if not os.path.exists(file):
        os.makedirs('data', exist_ok=True)
        if file == "data/users.json":
            data = {"users": {}}
        elif file == "data/signals.json":
            data = {"signals": []}
        elif file == "data/results.json":
            data = {"results": []}
        else:
            data = {}
        save_json(file, data)
        return data
    
    try:
        with open(file, "r", encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {file}: {e}. Creating new file.")
        if file == "data/users.json":
            data = {"users": {}}
        elif file == "data/signals.json":
            data = {"signals": []}
        elif file == "data/results.json":
            data = {"results": []}
        else:
            data = {}
        save_json(file, data)
        return data

def save_json(file, data):
    """Save JSON file with atomic write"""
    temp_file = f"{file}.tmp"
    try:
        with open(temp_file, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(temp_file, file)
    except Exception as e:
        print(f"Error saving {file}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Load data
users_data = load_json("data/users.json")
signals_data = load_json("data/signals.json")
results_data = load_json("data/results.json")

# =============================================================================
# AUTHENTICATION & AUTHORIZATION
# =============================================================================

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def is_admin():
    """Check if current user is admin"""
    return session.get('username') == ADMIN_USERNAME

def login_required(f):
    """Decorator for requiring login"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator for requiring admin privileges"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session or session.get('username') != ADMIN_USERNAME:
            flash('Bạn cần quyền admin để truy cập trang này.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# =============================================================================
# BINANCE API & INDICATORS
# =============================================================================

def get_klines(symbol, max_retries=3):
    """Fetch klines from Binance Futures API"""
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            return df
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    
    return None

def add_indicators(df):
    """Add technical indicators to dataframe"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # EMAs
    df["ema8"] = EMAIndicator(close, window=8).ema_indicator()
    df["ema21"] = EMAIndicator(close, window=21).ema_indicator()
    df["ema50"] = EMAIndicator(close, window=50).ema_indicator()
    df["ema200"] = EMAIndicator(close, window=200).ema_indicator()
    
    # MACD
    macd = MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    
    # RSI
    df["rsi14"] = RSIIndicator(close, window=14).rsi()
    
    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    
    # ATR
    atr = AverageTrueRange(high, low, close, window=14)
    df["atr"] = atr.average_true_range()
    
    # Keltner Channel
    typical_price = (high + low + close) / 3
    df["kc_mid"] = typical_price.rolling(20).mean()
    df["kc_range"] = df["atr"] * 1.5
    df["kc_upper"] = df["kc_mid"] + df["kc_range"]
    df["kc_lower"] = df["kc_mid"] - df["kc_range"]
    
    # VWAP
    df["vwap"] = (typical_price * volume).cumsum() / volume.cumsum()
    
    # Volume MA
    df["volume_ma20"] = volume.rolling(20).mean()
    
    # FVG Detection
    df["fvg_bull"] = (df["low"].shift(2) > df["high"].shift(1))
    df["fvg_bear"] = (df["high"].shift(2) < df["low"].shift(1))
    
    # Wick and Body
    df["body"] = abs(df["open"] - df["close"])
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    
    return df

# =============================================================================
# TRADING COMBOS (10 COMBO CHÍNH)
# =============================================================================

def combo1_fvg_squeeze_pro(df):
    """FVG Squeeze Pro - Swing Trade"""
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        squeeze = (last.bb_width < SQUEEZE_THRESHOLD and 
                  last.bb_upper < last.kc_upper and 
                  last.bb_lower > last.kc_lower)
        breakout_up = last.close > last.bb_upper and prev.close <= prev.bb_upper
        vol_spike = last.volume > last.volume_ma20 * 1.3
        trend_up = last.close > last.ema200
        rsi_ok = last.rsi14 < 68
        
        if squeeze and breakout_up and vol_spike and trend_up and rsi_ok:
            entry = last.close
            sl = entry - 1.5 * last.atr
            tp = entry + 3.0 * last.atr
            return "LONG", entry, sl, tp, "FVG Squeeze Pro", "SWING"
        
        breakout_down = last.close < last.bb_lower and prev.close >= prev.bb_lower
        if squeeze and breakout_down and vol_spike and last.close < last.ema200:
            entry = last.close
            sl = entry + 1.5 * last.atr
            tp = entry - 3.0 * last.atr
            return "SHORT", entry, sl, tp, "FVG Squeeze Pro", "SWING"
            
    except Exception as e:
        print(f"Combo1 error: {e}")
    return None

def combo2_macd_ob_retest(df):
    """MACD Order Block Retest - Swing Trade"""
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        macd_cross_up = last.macd > last.macd_signal and prev.macd <= prev.macd_signal
        price_above_ema200 = last.close > last.ema200
        
        ob_zone = None
        if all(df["close"].iloc[-3:] > df["open"].iloc[-3:]):
            ob_zone = df["low"].iloc[-5:-2].min()
        
        retest = ob_zone is not None and last.low <= ob_zone + last.atr * 0.5
        vol_confirm = last.volume > df["volume"].mean() * 1.1
        
        if macd_cross_up and price_above_ema200 and retest and vol_confirm:
            entry = last.close
            sl = ob_zone - last.atr
            tp = entry + 2.5 * last.atr
            return "LONG", entry, sl, tp, "MACD Order Block Retest", "SWING"
            
    except Exception as e:
        print(f"Combo2 error: {e}")
    return None

def combo3_vwap_ema_scalp(df):
    """VWAP + EMA Scalp - Scalping"""
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        ema_cross = last.ema8 > last.ema21 and prev.ema8 <= prev.ema21
        above_vwap = last.close > last.vwap
        vol_spike = last.volume > last.volume_ma20 * 1.8
        rsi_ok = last.rsi14 < 60
        
        if ema_cross and above_vwap and vol_spike and rsi_ok:
            entry = last.close
            sl = last.low - 0.5 * last.atr
            tp = entry + 1.0 * last.atr
            return "LONG", entry, sl, tp, "VWAP EMA Scalp", "SCALPING"
            
    except Exception as e:
        print(f"Combo3 error: {e}")
    return None

def combo4_rsi_extreme_bounce(df):
    """RSI Extreme Bounce - Intraday"""
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        rsi_oversold = last.rsi14 < 25
        rsi_overbought = last.rsi14 > 75
        
        bullish_engulfing = (last.close > last.open and 
                           prev.close < prev.open and 
                           last.close > prev.open and 
                           last.open < prev.close)
        
        bearish_engulfing = (last.close < last.open and 
                           prev.close > prev.open and 
                           last.close < prev.open and 
                           last.open > prev.close)
        
        hammer = (last.lower_wick > 2 * last.body and 
                last.upper_wick < 0.2 * last.body and 
                last.close > last.open) if last.body > 0 else False
                
        shooting_star = (last.upper_wick > 2 * last.body and 
                       last.lower_wick < 0.2 * last.body and 
                       last.close < last.open) if last.body > 0 else False
        
        vol_ok = last.volume > last.volume_ma20 * 1.2
        
        # LONG: RSI oversold + bullish pattern
        if rsi_oversold and (bullish_engulfing or hammer) and vol_ok:
            entry = last.close
            sl = last.low - 0.8 * last.atr
            tp = entry + 1.5 * last.atr
            return "LONG", entry, sl, tp, "RSI Extreme Bounce", "INTRADAY"
            
        # SHORT: RSI overbought + bearish pattern  
        if rsi_overbought and (bearish_engulfing or shooting_star) and vol_ok:
            entry = last.close
            sl = last.high + 0.8 * last.atr
            tp = entry - 1.5 * last.atr
            return "SHORT", entry, sl, tp, "RSI Extreme Bounce", "INTRADAY"
            
    except Exception as e:
        print(f"Combo4 error: {e}")
    return None

def combo5_ema_stack_volume(df):
    """EMA Stack + Volume - Intraday (COMBO MỚI)"""
    try:
        last = df.iloc[-1]
        
        ema_stack = (last.ema8 > last.ema21 > last.ema50 > last.ema200)
        price_above_all = (last.close > last.ema8 and 
                          last.close > last.ema21 and 
                          last.close > last.ema50 and 
                          last.close > last.ema200)
        volume_confirm = last.volume > last.volume_ma20 * 1.5
        rsi_ok = last.rsi14 < 65
        pullback_bounce = (
            (last.low <= last.ema8 and last.close > last.ema8) or
            (last.low <= last.ema21 and last.close > last.ema21)
        )
        
        if (ema_stack and price_above_all and volume_confirm and 
            rsi_ok and pullback_bounce):
            entry = last.close
            sl = min(last.ema21, last.low) - 0.3 * last.atr
            tp = entry + 1.8 * last.atr
            return "LONG", entry, sl, tp, "EMA Stack Volume", "INTRADAY"
            
    except Exception as e:
        print(f"Combo5 error: {e}")
    return None

def combo6_support_resistance_break(df):
    """Support/Resistance Break - Swing (COMBO MỚI)"""
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        resistance_level = df["high"].iloc[-20:-1].max()
        support_level = df["low"].iloc[-20:-1].min()
        
        resistance_break = (last.close > resistance_level and 
                           prev.close <= resistance_level)
        support_break = (last.close < support_level and 
                        prev.close >= support_level)
        volume_spike = last.volume > last.volume_ma20 * 1.8
        
        retest_confirmation = False
        if resistance_break:
            retest_confirmation = (last.low <= resistance_level and 
                                 last.close > resistance_level)
        elif support_break:
            retest_confirmation = (last.high >= support_level and 
                                 last.close < support_level)
        
        macd_confirm = (last.macd > last.macd_signal and 
                       last.macd_hist > 0)
        
        if ((resistance_break or support_break) and 
            volume_spike and retest_confirmation and macd_confirm):
            
            if resistance_break:
                entry = last.close
                sl = resistance_level - 0.5 * last.atr
                tp = entry + 2.0 * last.atr
                return "LONG", entry, sl, tp, "Support/Resistance Break", "SWING"
            else:
                entry = last.close  
                sl = support_level + 0.5 * last.atr
                tp = entry - 2.0 * last.atr
                return "SHORT", entry, sl, tp, "Support/Resistance Break", "SWING"
                
    except Exception as e:
        print(f"Combo6 error: {e}")
    return None

# Thêm các combo 7-10 tương tự...
def combo7_fvg_momentum(df):
    """FVG Momentum - Intraday"""
    try:
        last = df.iloc[-1]
        
        fvg = df["fvg_bull"].iloc[-2:].any() and last.close > last.open
        macd_mom = last.macd > last.macd_signal and abs(last.macd_hist) > abs(df["macd_hist"].iloc[-2])
        above_vwap = last.close > last.vwap
        low_vol = (last.atr / last.close) < 0.02
        
        if fvg and macd_mom and above_vwap and low_vol:
            entry = last.close
            sl = last.low - 0.5 * last.atr
            tp = entry + 1.2 * last.atr
            return "LONG", entry, sl, tp, "FVG Momentum", "INTRADAY"
            
    except Exception as e:
        print(f"Combo7 error: {e}")
    return None

def combo8_bb_squeeze_break(df):
    """BB Squeeze Break - Scalping"""
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        squeeze = last.bb_width < SQUEEZE_THRESHOLD
        breakout_up = last.close > last.bb_upper and prev.close <= prev.bb_upper
        volume_confirm = last.volume > last.volume_ma20 * 1.2
        
        if squeeze and breakout_up and volume_confirm:
            entry = last.close
            sl = last.low - 0.8 * last.atr
            tp = entry + 1.5 * last.atr
            return "LONG", entry, sl, tp, "BB Squeeze Break", "SCALPING"
            
    except Exception as e:
        print(f"Combo8 error: {e}")
    return None

def combo9_macd_divergence(df):
    """MACD Divergence - Swing"""
    try:
        last = df.iloc[-1]
        
        hist = df["macd_hist"]
        low = df["low"]
        
        divergence = hist.iloc[-1] > hist.iloc[-3] and low.iloc[-1] < low.iloc[-3]
        rsi_ok = last.rsi14 < 30
        
        if divergence and rsi_ok:
            entry = last.close
            sl = low.iloc[-5:].min() - last.atr
            tp = entry + 2.5 * last.atr
            return "LONG", entry, sl, tp, "MACD Divergence", "SWING"
            
    except Exception as e:
        print(f"Combo9 error: {e}")
    return None

def combo10_trend_pullback(df):
    """Trend Pullback - Intraday"""
    try:
        last = df.iloc[-1]
        
        trend_up = last.close > last.ema50 > last.ema200
        pullback = (last.low <= last.ema21 and last.close > last.ema21)
        volume_ok = last.volume > last.volume_ma20 * 1.1
        rsi_ok = 30 < last.rsi14 < 70
        
        if trend_up and pullback and volume_ok and rsi_ok:
            entry = last.close
            sl = last.ema21 - 0.5 * last.atr
            tp = entry + 2.0 * last.atr
            return "LONG", entry, sl, tp, "Trend Pullback", "INTRADAY"
            
    except Exception as e:
        print(f"Combo10 error: {e}")
    return None

# =============================================================================
# SCANNING & SIGNAL MANAGEMENT
# =============================================================================

def scan_signals():
    """Scan for trading signals"""
    print(f"[{datetime.now(timezone.utc)}] 🔍 Scanning signals...")
    
    combos = [
        combo1_fvg_squeeze_pro, combo2_macd_ob_retest, combo3_vwap_ema_scalp,
        combo4_rsi_extreme_bounce, combo5_ema_stack_volume, combo6_support_resistance_break,
        combo7_fvg_momentum, combo8_bb_squeeze_break, combo9_macd_divergence, combo10_trend_pullback
    ]
    
    new_signals = []
    
    for coin in COINS:
        df = get_klines(coin)
        if df is None or len(df) < 200:
            continue
            
        try:
            df = add_indicators(df)
        except Exception as e:
            print(f"Error adding indicators for {coin}: {e}")
            continue

        for combo_func in combos:
            try:
                result = combo_func(df)
                if result:
                    direction, entry, sl, tp, combo_name, signal_type = result
                    
                    # Check cooldown
                    signal_id = f"{coin}_{combo_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
                    
                    # Create signal object
                    signal = {
                        "id": signal_id,
                        "coin": coin,
                        "direction": direction,
                        "entry": float(entry),
                        "sl": float(sl),
                        "tp": float(tp),
                        "combo_name": combo_name,
                        "signal_type": signal_type,
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "active",  # active, win, lose
                        "win_lose": None
                    }
                    
                    new_signals.append(signal)
                    print(f"✅ Signal found: {coin} - {combo_name}")
                    
                    # Break after first signal per coin
                    break
                    
            except Exception as e:
                print(f"Error in {combo_func.__name__} for {coin}: {e}")
    
    # Save new signals
    if new_signals:
        with data_lock:
            current_signals = signals_data.get("signals", [])
            current_signals.extend(new_signals)
            signals_data["signals"] = current_signals
            save_json("data/signals.json", signals_data)
    
    print(f"✅ Scan complete. Found {len(new_signals)} new signals.")

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Home page - redirect to login"""
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check admin login
        if username == ADMIN_USERNAME and hash_password(password) == ADMIN_PASSWORD_HASH:
            session['username'] = username
            session['is_admin'] = True
            flash('Đăng nhập admin thành công!', 'success')
            return redirect(url_for('dashboard'))
        
        # Check user login
        with data_lock:
            users = users_data.get("users", {})
            if username in users and users[username]['password'] == hash_password(password):
                session['username'] = username
                session['is_admin'] = False
                flash('Đăng nhập thành công!', 'success')
                return redirect(url_for('dashboard'))
        
        flash('Tên đăng nhập hoặc mật khẩu không đúng!', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        email = request.form.get('email', '')
        
        if password != confirm_password:
            flash('Mật khẩu xác nhận không khớp!', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Mật khẩu phải có ít nhất 6 ký tự!', 'error')
            return render_template('register.html')
        
        with data_lock:
            users = users_data.get("users", {})
            if username in users:
                flash('Tên đăng nhập đã tồn tại!', 'error')
                return render_template('register.html')
            
            # Register new user
            users[username] = {
                'password': hash_password(password),
                'email': email,
                'created_at': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                'is_active': True
            }
            users_data["users"] = users
            save_json("data/users.json", users_data)
            
            flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    with data_lock:
        signals = signals_data.get("signals", [])
        # Get active signals (last 50 signals)
        active_signals = [s for s in signals if s.get('status') == 'active'][-50:]
        
        # Calculate stats
        total_signals = len(signals)
        active_count = len(active_signals)
        
        # Win/lose stats (only for admin or from results)
        win_count = len([s for s in signals if s.get('win_lose') == 'win'])
        lose_count = len([s for s in signals if s.get('win_lose') == 'lose'])
        
        # Group by type
        scalping_signals = [s for s in active_signals if s.get('signal_type') == 'SCALPING']
        intraday_signals = [s for s in active_signals if s.get('signal_type') == 'INTRADAY']
        swing_signals = [s for s in active_signals if s.get('signal_type') == 'SWING']
    
    return render_template('dashboard.html', 
                         signals=active_signals,
                         total_signals=total_signals,
                         active_count=active_count,
                         win_count=win_count,
                         lose_count=lose_count,
                         scalping_signals=scalping_signals,
                         intraday_signals=intraday_signals,
                         swing_signals=swing_signals,
                         is_admin=is_admin())

@app.route('/signals')
@login_required
def signals():
    """Signals management page"""
    signal_type = request.args.get('type', 'all')
    
    with data_lock:
        all_signals = signals_data.get("signals", [])
        
        if signal_type == 'scalping':
            signals_list = [s for s in all_signals if s.get('signal_type') == 'SCALPING']
        elif signal_type == 'intraday':
            signals_list = [s for s in all_signals if s.get('signal_type') == 'INTRADAY']
        elif signal_type == 'swing':
            signals_list = [s for s in all_signals if s.get('signal_type') == 'SWING']
        elif signal_type == 'active':
            signals_list = [s for s in all_signals if s.get('status') == 'active']
        else:
            signals_list = all_signals[-100:]  # Last 100 signals
        
        # Reverse to show newest first
        signals_list.reverse()
    
    return render_template('signals.html', 
                         signals=signals_list, 
                         signal_type=signal_type,
                         is_admin=is_admin())

@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin():
    """Admin panel for managing signals"""
    if request.method == 'POST':
        signal_id = request.form['signal_id']
        action = request.form['action']  # win or lose
        
        with data_lock:
            signals_list = signals_data.get("signals", [])
            for signal in signals_list:
                if signal['id'] == signal_id:
                    signal['win_lose'] = action
                    signal['status'] = 'closed'
                    signal['closed_at'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    break
            
            signals_data["signals"] = signals_list
            save_json("data/signals.json", signals_data)
            
            # Also save to results
            results = results_data.get("results", [])
            results.append({
                "signal_id": signal_id,
                "result": action,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            })
            results_data["results"] = results
            save_json("data/results.json", results_data)
        
        flash(f'Đã đánh dấu tín hiệu {signal_id} là {action.upper()}!', 'success')
        return redirect(url_for('admin'))
    
    with data_lock:
        active_signals = [s for s in signals_data.get("signals", []) if s.get('status') == 'active']
        closed_signals = [s for s in signals_data.get("signals", []) if s.get('status') == 'closed']
        users = users_data.get("users", {})
    
    return render_template('admin.html', 
                         active_signals=active_signals,
                         closed_signals=closed_signals[-50:],
                         total_users=len(users),
                         is_admin=True)

@app.route('/results')
@login_required
def results():
    """Results and performance page"""
    with data_lock:
        results_list = results_data.get("results", [])
        signals_list = signals_data.get("signals", [])
        
        # Combine results with signal details
        detailed_results = []
        for result in results_list[-100:]:  # Last 100 results
            signal_id = result['signal_id']
            signal = next((s for s in signals_list if s['id'] == signal_id), None)
            if signal:
                detailed_result = signal.copy()
                detailed_result.update(result)
                detailed_results.append(detailed_result)
        
        # Calculate performance
        win_count = len([r for r in results_list if r['result'] == 'win'])
        lose_count = len([r for r in results_list if r['result'] == 'lose'])
        total_trades = win_count + lose_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    return render_template('results.html',
                         results=detailed_results[::-1],  # Newest first
                         win_count=win_count,
                         lose_count=lose_count,
                         total_trades=total_trades,
                         win_rate=win_rate,
                         is_admin=is_admin())

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    with data_lock:
        users = users_data.get("users", {})
        user_data = users.get(session['username'], {})
    
    return render_template('profile.html',
                         username=session['username'],
                         user_data=user_data,
                         is_admin=is_admin())

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('Đã đăng xuất thành công!', 'success')
    return redirect(url_for('login'))

@app.route('/api/scan', methods=['POST'])
@admin_required
def api_scan():
    """API endpoint for manual scanning"""
    try:
        scan_signals()
        return jsonify({"status": "success", "message": "Scan completed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# =============================================================================
# INITIALIZATION & SCHEDULING
# =============================================================================

def initialize_app():
    """Initialize the application"""
    print("🚀 Initializing Trading Signals Website...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Start scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(scan_signals, 'interval', minutes=SCAN_INTERVAL_MINUTES)
    scheduler.start()
    print(f"✅ Scheduler started (interval: {SCAN_INTERVAL_MINUTES} minutes)")
    
    # Run initial scan
    print("🔍 Running initial scan...")
    try:
        scan_signals()
    except Exception as e:
        print(f"❌ Initial scan error: {e}")

# Initialize when imported
if __name__ == '__main__':
    initialize_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    initialize_app()
