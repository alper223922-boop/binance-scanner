"""
Futures Scanner - MEXC API (İyileştirilmiş)
- Trend filtreli sinyaller
- Ağırlıklı skorlama
- İndikatörler trende göre yorumlanır
- Minimum TP/SL koruması
- Dinamik fiyat formatı
"""

import asyncio
import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from telegram import Bot
from telegram.constants import ParseMode

from config import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TIMEFRAME, TOP_RESULTS,
    MIN_VOLUME_USDT, TREND_FILTER, MIN_SCORE, MIN_DIRECTION_SIGNALS,
    TP_SL, INDICATOR_WEIGHTS, EXCLUDED_SYMBOLS,
    get_tf_params, format_price,
)

# ===== LOG =====
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler  = logging.FileHandler("scanner.log", encoding="utf-8")
file_handler.setFormatter(log_formatter)
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
log = logging.getLogger(__name__)

MEXC_BASE = "https://contract.mexc.com"
HEADERS   = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# ============================================================
# MEXC API
# ============================================================
def get_symbols():
    try:
        r = requests.get(f"{MEXC_BASE}/api/v1/contract/detail", headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        symbols = [
            d["symbol"] for d in data
            if d.get("settleCoin") == "USDT"
            and d.get("state") == 0
            and d["symbol"].replace("_USDT", "") not in EXCLUDED_SYMBOLS
        ]
        log.info(f"Toplam sembol: {len(symbols)}")
        return symbols
    except Exception as e:
        log.error(f"Sembol listesi hatası: {e}")
        return []

def get_klines(symbol, interval="Min60", limit=200):
    try:
        url = f"{MEXC_BASE}/api/v1/contract/kline/{symbol}"
        r = requests.get(url, headers=HEADERS,
                         params={"interval": interval, "limit": limit}, timeout=10)
        if r.status_code != 200:
            return None
        raw = r.json().get("data", {})
        if not raw:
            return None
        df = pd.DataFrame({
            "open":  pd.to_numeric(raw.get("open", []),  errors="coerce"),
            "close": pd.to_numeric(raw.get("close", []), errors="coerce"),
            "high":  pd.to_numeric(raw.get("high", []),  errors="coerce"),
            "low":   pd.to_numeric(raw.get("low", []),   errors="coerce"),
            "vol":   pd.to_numeric(raw.get("vol", []),   errors="coerce"),
        })
        return df.dropna()
    except Exception as e:
        log.debug(f"{symbol} klines hatası: {e}")
        return None

def get_ticker(symbol):
    try:
        r = requests.get(f"{MEXC_BASE}/api/v1/contract/ticker?symbol={symbol}",
                         headers=HEADERS, timeout=5)
        if r.status_code != 200:
            return None
        return r.json().get("data", None)
    except:
        return None

# ============================================================
# İNDİKATÖR HESAPLAMALARI (ham değerler)
# ============================================================
def calc_rsi(close, p=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / (l + 1e-10)))

def calc_macd_full(close, fast=12, slow=26, sig=9):
    ml = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig, adjust=False).mean()
    hist = ml - sl
    return ml, sl, hist

def calc_bb(close, p=20, s=2):
    sma = close.rolling(p).mean()
    sd  = close.rolling(p).std()
    return sma + s*sd, sma - s*sd, sma

def calc_ema(close, p):
    return close.ewm(span=p, adjust=False).mean()

def calc_atr(high, low, close, p=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_cci(high, low, close, p=20):
    tp  = (high+low+close)/3
    sma = tp.rolling(p).mean()
    mad = tp.rolling(p).apply(lambda x: np.abs(x-x.mean()).mean())
    return (tp-sma)/(0.015*mad+1e-10)

def calc_wr(high, low, close, p=14):
    return -100*(high.rolling(p).max()-close)/(high.rolling(p).max()-low.rolling(p).min()+1e-10)

def calc_cmf(high, low, close, vol, period=20):
    mfv = ((close - low) - (high - close)) / (high - low + 1e-10) * vol
    return mfv.rolling(period).sum() / (vol.rolling(period).sum() + 1e-10)

def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    atr = calc_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    direction = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
    return direction

def calc_wavetrend(high, low, close, n1=10, n2=21):
    hlc3 = (high + low + close) / 3
    esa  = hlc3.ewm(span=n1, adjust=False).mean()
    d    = (hlc3 - esa).abs().ewm(span=n1, adjust=False).mean()
    ci   = (hlc3 - esa) / (0.015 * d + 1e-10)
    wt1  = ci.ewm(span=n2, adjust=False).mean()
    wt2  = wt1.rolling(4).mean()
    return wt1, wt2

def calc_squeeze(high, low, close, vol, bb_period=20, kc_mult=1.5):
    bb_mid   = close.rolling(bb_period).mean()
    bb_std   = close.rolling(bb_period).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    kc_atr   = calc_atr(high, low, close, bb_period)
    kc_upper = bb_mid + kc_mult * kc_atr
    kc_lower = bb_mid - kc_mult * kc_atr
    squeeze  = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    delta    = close - (high.rolling(bb_period).max() + low.rolling(bb_period).min()) / 2
    momentum = delta.rolling(bb_period).mean()
    return squeeze.iloc[-1], momentum.iloc[-1], momentum.iloc[-2]

def calc_ichimoku(high, low, close):
    tenkan   = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun    = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    price    = close.iloc[-1]
    cloud_top = max(senkou_a.iloc[-1], senkou_b.iloc[-1])
    cloud_bot = min(senkou_a.iloc[-1], senkou_b.iloc[-1])
    if price > cloud_top:   return "long",    "Above"
    elif price < cloud_bot: return "short",   "Below"
    return "neutral", "Inside"

def calc_fib_levels(high, low, close, period=50):
    hh = high.rolling(period).max().iloc[-1]
    ll = low.rolling(period).min().iloc[-1]
    price = close.iloc[-1]
    diff = hh - ll
    if diff == 0:
        return {}, price, "0.5", 999
    levels = {
        "0.236": hh - 0.236 * diff,
        "0.382": hh - 0.382 * diff,
        "0.500": hh - 0.500 * diff,
        "0.618": hh - 0.618 * diff,
        "0.786": hh - 0.786 * diff,
    }
    pct_pos = (price - ll) / diff  # 0=dipte, 1=zirvede
    closest = min(levels, key=lambda k: abs(levels[k] - price) / price)
    closest_price = levels[closest]
    distance_pct = abs(price - closest_price) / price * 100
    return levels, pct_pos, closest, distance_pct

# ============================================================
# SİNYAL YORUMLAYICILARI (trend yönüne göre)
# ============================================================
def sig_macd(macd_line, signal_line, hist, prev_hist, trend):
    macd_above = macd_line > signal_line
    hist_rising = hist > prev_hist
    hist_pos = hist > 0

    if macd_above and hist_rising and hist_pos:
        return "long"
    if (not macd_above) and (not hist_rising) and (not hist_pos):
        return "short"
    if trend == "long" and macd_above:
        return "long"
    if trend == "short" and (not macd_above):
        return "short"
    return "neutral"

def sig_rsi(rsi_val, trend):
    if trend == "long":
        if rsi_val < 30:    return "long"
        if rsi_val > 75:    return "short"
        if 40 <= rsi_val <= 60: return "long"
    elif trend == "short":
        if rsi_val > 70:    return "short"
        if rsi_val < 25:    return "long"
        if 40 <= rsi_val <= 60: return "short"
    return "neutral"

def sig_cci(cci_val, trend):
    if trend == "long":
        if cci_val < -100: return "long"
        if cci_val > 200:  return "short"
        if -100 <= cci_val <= 100: return "long"
    elif trend == "short":
        if cci_val > 100:  return "short"
        if cci_val < -200: return "long"
        if -100 <= cci_val <= 100: return "short"
    return "neutral"

def sig_wr(wr_val, trend):
    if trend == "long":
        if wr_val < -80: return "long"
        if wr_val > -20: return "short"
    elif trend == "short":
        if wr_val > -20: return "short"
        if wr_val < -80: return "long"
    return "neutral"

def sig_cmf(cmf_val, trend):
    if cmf_val > 0.15:  return "long"
    if cmf_val < -0.15: return "short"
    if cmf_val > 0.05 and trend == "long":  return "long"
    if cmf_val < -0.05 and trend == "short": return "short"
    return "neutral"

def sig_bb(price, bb_upper, bb_lower, bb_mid, trend):
    bb_range = bb_upper - bb_lower
    pct_b = (price - bb_lower) / (bb_range + 1e-10)
    if pct_b < 0.05:
        return "long" if trend != "short" else "neutral"
    if pct_b > 0.95:
        return "short" if trend != "long" else "neutral"
    if trend == "long" and price > bb_mid:  return "long"
    if trend == "short" and price < bb_mid: return "short"
    return "neutral"

def sig_supertrend(dir_series):
    if len(dir_series) < 3:
        return "neutral"
    c, p1, p2 = dir_series.iloc[-1], dir_series.iloc[-2], dir_series.iloc[-3]
    if c == 1 and p1 == 1 and p2 == 1:   return "long"
    if c == -1 and p1 == -1 and p2 == -1: return "short"
    return "neutral"

def sig_wavetrend(wt1_val, wt2_val, trend):
    if wt1_val < -60 and wt1_val > wt2_val: return "long"
    if wt1_val > 60  and wt1_val < wt2_val: return "short"
    if trend == "long"  and wt1_val > wt2_val and wt1_val > -20: return "long"
    if trend == "short" and wt1_val < wt2_val and wt1_val < 20:  return "short"
    return "neutral"

def sig_fib(pct_pos, distance_pct, trend):
    # Seviyeye %1.5'ten uzaksa sinyal verme
    if distance_pct > 1.5:
        return "neutral"
    # Destek bölgesi (0.5-0.786) + trend yukarı = long
    if pct_pos >= 0.5 and trend == "long":
        return "long"
    # Direnç bölgesi (0.236-0.5) + trend aşağı = short
    if pct_pos <= 0.5 and trend == "short":
        return "short"
    return "neutral"

def sig_squeeze(sq_on, sq_mom, sq_prev, trend):
    if sq_on:
        return "neutral"  # Sıkışma = belirsizlik
    if sq_mom > 0 and sq_mom > sq_prev and trend == "long":  return "long"
    if sq_mom < 0 and sq_mom < sq_prev and trend == "short": return "short"
    return "neutral"

def dot(s):
    return "🟢" if s == "long" else ("🔴" if s == "short" else "🟡")

# ============================================================
# ANA ANALİZ
# ============================================================
def analyze(symbol, df, ticker):
    p = get_tf_params()
    if df is None or len(df) < p["min_candles"]:
        return None

    try:
        close, high, low, vol = df["close"], df["high"], df["low"], df["vol"]
        price = close.iloc[-1]
    except Exception as e:
        log.debug(f"{symbol} df hatası: {e}")
        return None

    # ---------- Ham indikatör değerleri ----------
    macd_line, sig_line, hist = calc_macd_full(close, p["macd_fast"], p["macd_slow"], p["macd_sig"])
    macd_v, sig_v = macd_line.iloc[-1], sig_line.iloc[-1]
    hist_v, prev_hist_v = hist.iloc[-1], hist.iloc[-2]

    bb_upper, bb_lower, bb_mid = calc_bb(close, p["bb_p"])
    bb_u, bb_l, bb_m = bb_upper.iloc[-1], bb_lower.iloc[-1], bb_mid.iloc[-1]

    ema_fast = calc_ema(close, p["ema_fast"]).iloc[-1]
    ema_slow = calc_ema(close, p["ema_slow"]).iloc[-1]

    atr_v = calc_atr(high, low, close, p["atr_p"]).iloc[-1]

    cci_v = calc_cci(high, low, close, p["cci_p"]).iloc[-1]
    wr_v  = calc_wr(high, low, close, p["wr_p"]).iloc[-1]
    cmf_v = calc_cmf(high, low, close, vol, p["cmf_p"]).iloc[-1]
    rsi_v = calc_rsi(close, p["rsi_p"]).iloc[-1]

    st_dir = calc_supertrend(high, low, close, p["st_p"], p["st_mult"])
    ich_sig, ich_pos = calc_ichimoku(high, low, close)

    wt1, wt2 = calc_wavetrend(high, low, close, p["wt_n1"], p["wt_n2"])
    wt1_v, wt2_v = wt1.iloc[-1], wt2.iloc[-1]

    sq_on, sq_mom, sq_prev = calc_squeeze(high, low, close, vol, p["bb_p"], p["sq_kc_mult"])
    fib_levels, fib_pct_pos, fib_closest, fib_dist = calc_fib_levels(high, low, close, p["fib_p"])

    # ---------- Önce TREND yönünü belirle ----------
    st_sig = sig_supertrend(st_dir)
    ema_sig = "long" if ema_fast > ema_slow and price > ema_fast else (
              "short" if ema_fast < ema_slow and price < ema_fast else "neutral")

    trend_votes = [st_sig, ich_sig, ema_sig]
    long_votes  = sum(1 for x in trend_votes if x == "long")
    short_votes = sum(1 for x in trend_votes if x == "short")

    if long_votes >= TREND_FILTER["min_aligned"]:
        trend = "long"
    elif short_votes >= TREND_FILTER["min_aligned"]:
        trend = "short"
    else:
        trend = "neutral"

    # Trend belirsizse hiç sinyal verme
    if TREND_FILTER["enabled"] and trend == "neutral":
        return None

    # ---------- Şimdi diğer indikatörleri trende göre yorumla ----------
    ind = {}
    weighted_score = 0.0
    long_signals = 0
    short_signals = 0

    def add(name, val, sig):
        nonlocal weighted_score, long_signals, short_signals
        ind[name] = {"value": val, "signal": sig}
        weight = INDICATOR_WEIGHTS.get(name.split("(")[0], 1.0)
        if sig == "long":
            weighted_score += weight
            long_signals += 1
        elif sig == "short":
            weighted_score -= weight
            short_signals += 1

    add("MACD",     f"{hist_v:.5f}", sig_macd(macd_v, sig_v, hist_v, prev_hist_v, trend))
    add("BB%",      f"{(price-bb_l)/(bb_u-bb_l+1e-10)*100:.1f}%", sig_bb(price, bb_u, bb_l, bb_m, trend))
    add("EMA",      f"{ema_fast/ema_slow:.4f}", ema_sig)
    add("ATR%",     f"{atr_v/price*100:.2f}%", "neutral")
    add("CCI",      f"{cci_v:.0f}",   sig_cci(cci_v, trend))
    add("W%R",      f"{wr_v:.0f}",    sig_wr(wr_v, trend))
    add("Fib",      f"Lvl:{fib_closest}", sig_fib(fib_pct_pos, fib_dist, trend))
    add("CMF",      f"{cmf_v:.3f}",   sig_cmf(cmf_v, trend))
    add(f"RSI({p['rsi_p']})", f"{rsi_v:.0f}", sig_rsi(rsi_v, trend))
    add("Supertrend", "Buy" if st_sig=="long" else ("Sell" if st_sig=="short" else "Idle"), st_sig)
    add("Ichimoku",  ich_pos, ich_sig)
    add("WaveTrend", f"{wt1_v:.0f}",  sig_wavetrend(wt1_v, wt2_v, trend))
    add("Squeeze",   "Squeezing" if sq_on else (
                     "Mom Up" if sq_mom>0 and sq_mom>sq_prev else (
                     "Mom Dn" if sq_mom<0 and sq_mom<sq_prev else "Idle")),
                     sig_squeeze(sq_on, sq_mom, sq_prev, trend))

    # ---------- Yön ve minimum eşik kontrolleri ----------
    if long_signals >= MIN_DIRECTION_SIGNALS and weighted_score >= MIN_SCORE:
        direction = "long"
    elif short_signals >= MIN_DIRECTION_SIGNALS and weighted_score <= -MIN_SCORE:
        direction = "short"
    else:
        return None

    # Trend ile yön uyumlu mu?
    if direction != trend:
        return None

    # ---------- TP/SL (minimum yüzde koruması) ----------
    def make_levels(direction):
        if direction == "long":
            tp1 = price + max(atr_v * TP_SL["tp1_atr_mult"], price * TP_SL["min_tp1_percent"]/100)
            tp2 = price + max(atr_v * TP_SL["tp2_atr_mult"], price * 2*TP_SL["min_tp1_percent"]/100)
            tp3 = price + max(atr_v * TP_SL["tp3_atr_mult"], price * TP_SL["min_tp3_percent"]/100)
            sl  = price - min(atr_v * TP_SL["sl_atr_mult"],  price * TP_SL["max_sl_percent"]/100)
        else:
            tp1 = price - max(atr_v * TP_SL["tp1_atr_mult"], price * TP_SL["min_tp1_percent"]/100)
            tp2 = price - max(atr_v * TP_SL["tp2_atr_mult"], price * 2*TP_SL["min_tp1_percent"]/100)
            tp3 = price - max(atr_v * TP_SL["tp3_atr_mult"], price * TP_SL["min_tp3_percent"]/100)
            sl  = price + min(atr_v * TP_SL["sl_atr_mult"],  price * TP_SL["max_sl_percent"]/100)
        return tp1, tp2, tp3, sl

    tp1_l, tp2_l, tp3_l, sl_l = make_levels("long")
    tp1_s, tp2_s, tp3_s, sl_s = make_levels("short")

    # ---------- 24h değişim ----------
    try:
        pc   = float(ticker.get("riseFallRate", 0)) * 100
        v24  = float(ticker.get("amount24", ticker.get("volume24", 0)))
        v24h = float(ticker.get("volume24", 0))
        v24h_prev = v24h / (1 + abs(pc) / 100 + 0.001)
        vol_change = ((v24h - v24h_prev) / (v24h_prev + 1)) * 100
    except:
        pc = 0; v24 = 0; vol_change = 0

    return {
        "symbol": symbol.replace("_USDT", ""),
        "price": price,
        "direction": direction,
        "trend": trend,
        "weighted_score": weighted_score,
        "long_count": long_signals,
        "short_count": short_signals,
        "indicators": ind,
        "tp1_long": tp1_l, "tp2_long": tp2_l, "tp3_long": tp3_l, "sl_long": sl_l,
        "tp1_short": tp1_s, "tp2_short": tp2_s, "tp3_short": tp3_s, "sl_short": sl_s,
        "price_change_24h": pc,
        "volume_24h": v24,
        "vol_change_24h": vol_change,
    }

# ============================================================
# MESAJ FORMATLAMA
# ============================================================
def fmt_vol(v):
    if v >= 1e9: return f"${v/1e9:.2f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def fmt_pct(val, ref):
    pct = (val - ref) / ref * 100
    return f"{format_price(val)} ({pct:+.2f}%)"

def fmt_coin_msg(r, direction, rank, total):
    price = r["price"]
    pc    = r["price_change_24h"]
    vc    = r.get("vol_change_24h", 0)
    ind   = r["indicators"]

    if direction == "long":
        arrow = "🚀 LONG"
        tp1, tp2, tp3, sl = r["tp1_long"], r["tp2_long"], r["tp3_long"], r["sl_long"]
        cnt = r["long_count"]
        color = "🟢"
    else:
        arrow = "🔻 SHORT"
        tp1, tp2, tp3, sl = r["tp1_short"], r["tp2_short"], r["tp3_short"], r["sl_short"]
        cnt = r["short_count"]
        color = "🔴"

    sl_pct = abs(sl - price) / price * 100
    rr3 = abs((tp3 - price) / price * 100 / sl_pct) if sl_pct != 0 else 0
    fire = "🔥" * min(cnt, 5)
    lines = [f"  {dot(d['signal'])} {n}: {d['value']}" for n, d in ind.items()]

    msg = (
        f"{color} *{r['symbol']}* {arrow} {fire}\n"
        f"#{rank}/{total} | TF: `{TIMEFRAME}` | Trend: `{r['trend'].upper()}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 Fiyat: `${format_price(price)}`\n"
        f"📊 24h: {'📈' if pc>=0 else '📉'} {pc:+.2f}% | Vol: {fmt_vol(r['volume_24h'])} ({'📈' if vc>=0 else '📉'} {vc:+.1f}%)\n"
        f"📐 Ağırlıklı Skor: `{r['weighted_score']:+.1f}` ({cnt} net sinyal)\n"
        f"\n"
        f"🎯 *TP1:* `${fmt_pct(tp1, price)}`\n"
        f"🎯 *TP2:* `${fmt_pct(tp2, price)}`\n"
        f"🎯 *TP3:* `${fmt_pct(tp3, price)}`\n"
        f"🛑 *SL:*  `${fmt_pct(sl, price)}`\n"
        f"⚖️ R/R (TP3): 1:{rr3:.1f}\n"
        f"\n"
        f"📈 *Göstergeler*\n"
        + "\n".join(lines)
    )
    return msg

def build_header(direction, total):
    ts = datetime.utcnow().strftime("%d.%m.%Y %H:%M")
    if direction == "long":
        return f"🚀━━ TOP {total} LONG ━━🚀\n🤖 MEXC | {ts} UTC | `{TIMEFRAME}`"
    else:
        return f"🔻━━ TOP {total} SHORT ━━🔻\n🤖 MEXC | {ts} UTC | `{TIMEFRAME}`"

# ============================================================
# ANA ÇALIŞTIRICI
# ============================================================
async def run_scan():
    log.info(f"MEXC tarama başlıyor... TF={TIMEFRAME}")
    symbols = get_symbols()
    if not symbols:
        log.error("Sembol listesi boş!"); return

    results, failed = [], 0
    for i, sym in enumerate(symbols):
        try:
            df = get_klines(sym, TIMEFRAME, 200)
            if df is None: continue
            ticker = get_ticker(sym)
            if ticker is None: continue
            if float(ticker.get("amount24", 0)) < MIN_VOLUME_USDT: continue

            result = analyze(sym, df, ticker)
            if result:
                results.append(result)
                log.info(f"✓ {result['symbol']} → {result['direction'].upper()} "
                         f"(skor={result['weighted_score']:+.1f}, "
                         f"{result['long_count']}L/{result['short_count']}S)")

            time.sleep(1 if i % 50 == 0 and i > 0 else 0.05)
        except Exception as e:
            failed += 1
            if failed <= 3:
                log.error(f"{sym} HATA: {type(e).__name__}: {e}")
            else:
                log.debug(f"{sym}: {e}")

    log.info(f"Tamamlandı: {len(results)} geçerli, {failed} hata")
    if not results:
        log.warning("Sonuç bulunamadı!"); return

    long_results  = [r for r in results if r["direction"] == "long"]
    short_results = [r for r in results if r["direction"] == "short"]

    top_longs  = sorted(long_results,  key=lambda x: x["weighted_score"], reverse=True)[:TOP_RESULTS]
    top_shorts = sorted(short_results, key=lambda x: x["weighted_score"])[:TOP_RESULTS]

    log.info(f"Long: {len(top_longs)} | Short: {len(top_shorts)}")

    if "YOUR_TOKEN" in TELEGRAM_TOKEN or not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN ayarlanmamış!"); return

    bot = Bot(token=TELEGRAM_TOKEN)

    async def send(msg):
        try:
            sent = await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg,
                                          parse_mode=ParseMode.MARKDOWN)
            log.info(f"Mesaj OK id={sent.message_id}")
            await asyncio.sleep(1)
        except Exception as e:
            log.error(f"Telegram hatası: {e}")
            try:
                plain = msg.replace("*","").replace("`","").replace("_","")
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=plain)
                await asyncio.sleep(1)
            except Exception as e2:
                log.error(f"Plain de başarısız: {e2}")

    if top_longs:
        await send(build_header("long", len(top_longs)))
        for i, r in enumerate(top_longs):
            await send(fmt_coin_msg(r, "long", i+1, len(top_longs)))

    if top_shorts:
        await send(build_header("short", len(top_shorts)))
        for i, r in enumerate(top_shorts):
            await send(fmt_coin_msg(r, "short", i+1, len(top_shorts)))

    log.info("Tüm mesajlar gönderildi ✅")

if __name__ == "__main__":
    log.info("Scanner başlıyor...")
    asyncio.run(run_scan())
