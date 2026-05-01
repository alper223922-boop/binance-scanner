"""
Binance Futures Scanner - Top 10 Long/Short Sinyali
Telegram'a TP/SL ve 10 gösterge ile sinyal gönderir
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
import schedule
import threading
from dotenv import load_dotenv

# .env dosyasini yukle
load_dotenv()

# ─── YAPILANDIRMA ───────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL", "60"))
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
TOP_RESULTS = 10
MIN_VOLUME_USDT = float(os.getenv("MIN_VOLUME_USDT", "5000000"))

# ─── LOGGING (Windows UTF-8 uyumlu) ────────────────────────────
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

file_handler = logging.FileHandler("scanner.log", encoding="utf-8")
file_handler.setFormatter(log_formatter)

# Windows'ta konsol UTF-8 zorla (Python 3.7+)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
log = logging.getLogger(__name__)

BINANCE_BASE = "https://fapi.binance.com"

# ─── BİNANCE API ────────────────────────────────────────────────
def get_futures_symbols():
    """Tüm aktif USDT vadeli sembolleri al"""
    r = requests.get(f"{BINANCE_BASE}/fapi/v1/exchangeInfo", timeout=10)
    r.raise_for_status()
    symbols = [
        s["symbol"] for s in r.json()["symbols"]
        if s["status"] == "TRADING" and s["quoteAsset"] == "USDT"
    ]
    return symbols

def get_klines(symbol, interval="1h", limit=200):
    """OHLCV verisi çek"""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(f"{BINANCE_BASE}/fapi/v1/klines", params=params, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    for col in ["open","high","low","close","volume","quote_vol"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def get_ticker_24h(symbol):
    """24 saatlik fiyat ve hacim değişimi"""
    r = requests.get(f"{BINANCE_BASE}/fapi/v1/ticker/24hr", params={"symbol": symbol}, timeout=5)
    if r.status_code != 200:
        return None
    return r.json()

# ─── TEKNİK GÖSTERGELER ─────────────────────────────────────────

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bollinger(close, period=20, std_dev=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def calc_ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def calc_stochastic(high, low, close, k=14, d=3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def calc_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad + 1e-10)

def calc_williams_r(high, low, close, period=14):
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0
    
    tr = calc_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / (tr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (tr + 1e-10))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

# ─── GÖSTERGE YORUMLAMA ─────────────────────────────────────────
def signal_dot(signal):
    """🟢 = long/bullish, 🔴 = short/bearish, 🟡 = nötr"""
    if signal == "long":   return "🟢"
    if signal == "short":  return "🔴"
    return "🟡"

def analyze_symbol(symbol, df, ticker):
    """10 gösterge hesapla, long/short skoru üret"""
    if df is None or len(df) < 50:
        return None

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]
    
    # Fiyat bilgileri
    price = close.iloc[-1]
    
    # ── 10 GÖSTERGE ──
    indicators = {}
    scores = []   # +1 long, -1 short, 0 nötr
    
    # 1. RSI (14)
    rsi = calc_rsi(close).iloc[-1]
    if rsi < 35:      rsi_sig = "long"
    elif rsi > 65:    rsi_sig = "short"
    else:             rsi_sig = "neutral"
    indicators["RSI(14)"] = {"value": f"{rsi:.1f}", "signal": rsi_sig}
    scores.append(1 if rsi_sig == "long" else (-1 if rsi_sig == "short" else 0))

    # 2. MACD
    macd_line, macd_signal, macd_hist = calc_macd(close)
    hist_val = macd_hist.iloc[-1]
    hist_prev = macd_hist.iloc[-2]
    if hist_val > 0 and hist_val > hist_prev:   macd_sig = "long"
    elif hist_val < 0 and hist_val < hist_prev: macd_sig = "short"
    else:                                        macd_sig = "neutral"
    indicators["MACD"] = {"value": f"{hist_val:.4f}", "signal": macd_sig}
    scores.append(1 if macd_sig == "long" else (-1 if macd_sig == "short" else 0))

    # 3. Bollinger Band
    bb_upper, bb_mid, bb_lower = calc_bollinger(close)
    p = close.iloc[-1]
    if p < bb_lower.iloc[-1]:   bb_sig = "long"
    elif p > bb_upper.iloc[-1]: bb_sig = "short"
    else:                        bb_sig = "neutral"
    bb_pct = ((p - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100)
    indicators["BB%"] = {"value": f"{bb_pct:.1f}%", "signal": bb_sig}
    scores.append(1 if bb_sig == "long" else (-1 if bb_sig == "short" else 0))

    # 4. EMA 20/50 Kesişimi
    ema20 = calc_ema(close, 20).iloc[-1]
    ema50 = calc_ema(close, 50).iloc[-1]
    if ema20 > ema50 and close.iloc[-1] > ema20:   ema_sig = "long"
    elif ema20 < ema50 and close.iloc[-1] < ema20: ema_sig = "short"
    else:                                            ema_sig = "neutral"
    indicators["EMA20/50"] = {"value": f"{ema20/ema50:.4f}", "signal": ema_sig}
    scores.append(1 if ema_sig == "long" else (-1 if ema_sig == "short" else 0))

    # 5. Stochastic RSI
    stoch_k, stoch_d = calc_stochastic(high, low, close)
    sk = stoch_k.iloc[-1]
    sd = stoch_d.iloc[-1]
    if sk < 20 and sk > sd:      sto_sig = "long"
    elif sk > 80 and sk < sd:    sto_sig = "short"
    else:                         sto_sig = "neutral"
    indicators["Stoch"] = {"value": f"K:{sk:.0f} D:{sd:.0f}", "signal": sto_sig}
    scores.append(1 if sto_sig == "long" else (-1 if sto_sig == "short" else 0))

    # 6. ATR (Volatilite)
    atr = calc_atr(high, low, close).iloc[-1]
    atr_pct = atr / price * 100
    # ATR trend göstergesi değil, TP/SL için kullanılacak - nötr sinyalle ekle
    indicators["ATR%"] = {"value": f"{atr_pct:.2f}%", "signal": "neutral"}
    # ATR skora dahil etmiyoruz, sadece bilgi

    # 7. OBV Trend
    obv = calc_obv(close, vol)
    obv_ema = obv.ewm(span=20).mean()
    if obv.iloc[-1] > obv_ema.iloc[-1]:    obv_sig = "long"
    elif obv.iloc[-1] < obv_ema.iloc[-1]:  obv_sig = "short"
    else:                                   obv_sig = "neutral"
    indicators["OBV"] = {"value": "↑" if obv_sig == "long" else ("↓" if obv_sig == "short" else "→"), "signal": obv_sig}
    scores.append(1 if obv_sig == "long" else (-1 if obv_sig == "short" else 0))

    # 8. CCI (20)
    cci = calc_cci(high, low, close).iloc[-1]
    if cci < -100:   cci_sig = "long"
    elif cci > 100:  cci_sig = "short"
    else:             cci_sig = "neutral"
    indicators["CCI(20)"] = {"value": f"{cci:.0f}", "signal": cci_sig}
    scores.append(1 if cci_sig == "long" else (-1 if cci_sig == "short" else 0))

    # 9. Williams %R
    wr = calc_williams_r(high, low, close).iloc[-1]
    if wr < -80:   wr_sig = "long"
    elif wr > -20: wr_sig = "short"
    else:           wr_sig = "neutral"
    indicators["W%R"] = {"value": f"{wr:.0f}", "signal": wr_sig}
    scores.append(1 if wr_sig == "long" else (-1 if wr_sig == "short" else 0))

    # 10. ADX (Trend Gücü)
    adx, plus_di, minus_di = calc_adx(high, low, close)
    adx_val = adx.iloc[-1]
    pdi = plus_di.iloc[-1]
    mdi = minus_di.iloc[-1]
    if adx_val > 25 and pdi > mdi:    adx_sig = "long"
    elif adx_val > 25 and mdi > pdi:  adx_sig = "short"
    else:                               adx_sig = "neutral"
    indicators["ADX"] = {"value": f"{adx_val:.0f}", "signal": adx_sig}
    scores.append(1 if adx_sig == "long" else (-1 if adx_sig == "short" else 0))

    # Toplam skor
    total_score = sum(scores)
    long_count  = scores.count(1)
    short_count = scores.count(-1)

    # TP/SL hesapla (ATR tabanlı)
    atr_val = atr  # son ATR değeri
    tp_long  = price + 2.5 * atr_val
    sl_long  = price - 1.5 * atr_val
    tp_short = price - 2.5 * atr_val
    sl_short = price + 1.5 * atr_val

    # 24h değişimler
    price_change_24h = float(ticker["priceChangePercent"]) if ticker else 0
    volume_24h = float(ticker["quoteVolume"]) if ticker else 0
    volume_prev = volume_24h / (1 + abs(price_change_24h) / 100 + 0.001)  # yaklaşık
    vol_change_24h = ((volume_24h - volume_prev) / (volume_prev + 1)) * 100

    return {
        "symbol": symbol,
        "price": price,
        "score": total_score,
        "long_count": long_count,
        "short_count": short_count,
        "indicators": indicators,
        "tp_long": tp_long,
        "sl_long": sl_long,
        "tp_short": tp_short,
        "sl_short": sl_short,
        "atr_val": atr_val,
        "price_change_24h": price_change_24h,
        "volume_24h": volume_24h,
        "vol_change_24h": vol_change_24h,
    }

# ─── MESAJ FORMATLAMA ───────────────────────────────────────────
def format_volume(v):
    if v >= 1_000_000_000: return f"${v/1e9:.2f}B"
    if v >= 1_000_000:     return f"${v/1e6:.1f}M"
    if v >= 1_000:         return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def format_signal_block(result, direction):
    sym = result["symbol"]
    price = result["price"]
    ind = result["indicators"]
    
    pc = result["price_change_24h"]
    pc_str = f"{'🟢' if pc >= 0 else '🔴'} {pc:+.2f}%"
    
    vc = result["vol_change_24h"]
    vc_str = f"{'📈' if vc >= 0 else '📉'} {vc:+.1f}%"
    vol_str = format_volume(result["volume_24h"])
    
    if direction == "long":
        arrow = "🚀 LONG"
        tp = result["tp_long"]
        sl = result["sl_long"]
        tp_pct = (tp - price) / price * 100
        sl_pct = (sl - price) / price * 100
    else:
        arrow = "🔻 SHORT"
        tp = result["tp_short"]
        sl = result["sl_short"]
        tp_pct = (tp - price) / price * 100
        sl_pct = (sl - price) / price * 100

    # Gösterge satırları
    ind_lines = []
    for name, data in ind.items():
        dot = signal_dot(data["signal"])
        ind_lines.append(f"  {dot} {name}: {data['value']}")
    
    rr = abs(tp_pct / sl_pct) if sl_pct != 0 else 0
    score_str = f"{'🔥' * min(result['long_count'] if direction=='long' else result['short_count'], 5)}"
    
    msg = (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"*{sym}* {arrow} {score_str}\n"
        f"💵 Fiyat: `${price:.4f}`\n"
        f"📊 24h Değişim: {pc_str}\n"
        f"💹 Hacim: {vol_str} | Değ: {vc_str}\n"
        f"\n"
        f"🎯 TP: `${tp:.4f}` ({tp_pct:+.2f}%)\n"
        f"🛑 SL: `${sl:.4f}` ({sl_pct:+.2f}%)\n"
        f"⚖️ R/R: 1:{rr:.1f}\n"
        f"\n"
        f"📈 *Göstergeler*\n"
    )
    msg += "\n".join(ind_lines)
    msg += "\n"
    return msg

def build_telegram_message(longs, shorts, timeframe):
    ts = datetime.now().strftime("%d.%m.%Y %H:%M")
    header = (
        f"🤖 *Binance Futures Tarama*\n"
        f"⏰ {ts} | TF: `{timeframe}`\n"
        f"🟢 Sinyal: Bullish  🔴 Bearish  🟡 Nötr\n"
    )
    
    long_header = "\n🚀━━━━━ TOP 10 LONG ━━━━━🚀\n"
    long_blocks = [format_signal_block(r, "long") for r in longs]
    
    short_header = "\n🔻━━━━━ TOP 10 SHORT ━━━━━🔻\n"
    short_blocks = [format_signal_block(r, "short") for r in shorts]
    
    footer = f"\n_Göstergeler: RSI · MACD · BB · EMA · Stoch · ATR · OBV · CCI · W%R · ADX_"
    
    # Telegram 4096 karakter limiti - parçalara böl
    messages = []
    
    # Mesaj 1: Header + Longlar
    m1 = header + long_header + "\n".join(long_blocks)
    messages.append(m1)
    
    # Mesaj 2: Shortlar
    m2 = short_header + "\n".join(short_blocks) + footer
    messages.append(m2)
    
    return messages

# ─── ANA TARAMA FONKSİYONU ──────────────────────────────────────
async def run_scan():
    log.info("Tarama basliyor...")

    try:
        symbols = get_futures_symbols()
        log.info(f"{len(symbols)} sembol bulundu")
    except Exception as e:
        log.error(f"Sembol listesi alinamadi: {e}")
        return

    results = []
    failed = 0

    for i, sym in enumerate(symbols):
        try:
            df = get_klines(sym, TIMEFRAME, 200)
            ticker = get_ticker_24h(sym)

            if ticker is None or float(ticker.get("quoteVolume", 0)) < MIN_VOLUME_USDT:
                continue

            result = analyze_symbol(sym, df, ticker)
            if result:
                results.append(result)

            # Rate limit koruma
            if i % 50 == 0 and i > 0:
                log.info(f"  {i}/{len(symbols)} tarandi...")
                time.sleep(1)
            else:
                time.sleep(0.05)

        except Exception as e:
            failed += 1
            log.debug(f"{sym} hata: {e}")
            continue

    log.info(f"Tarama tamamlandi: {len(results)} gecerli, {failed} hata")

    if not results:
        log.warning("Hic sonuc bulunamadi!")
        return

    # Sirala
    top_longs  = sorted(results, key=lambda x: x["score"], reverse=True)[:TOP_RESULTS]
    top_shorts = sorted(results, key=lambda x: x["score"])[:TOP_RESULTS]

    log.info(f"En iyi long : {top_longs[0]['symbol']}  skor={top_longs[0]['score']}")
    log.info(f"En iyi short: {top_shorts[0]['symbol']} skor={top_shorts[0]['score']}")

    # Token / Chat ID kontrol
    if "YOUR_TELEGRAM" in TELEGRAM_TOKEN or not TELEGRAM_TOKEN:
        log.error("HATA: .env dosyasinda TELEGRAM_TOKEN ayarlanmamis!")
        return
    if "YOUR_CHAT" in TELEGRAM_CHAT_ID or not TELEGRAM_CHAT_ID:
        log.error("HATA: .env dosyasinda TELEGRAM_CHAT_ID ayarlanmamis!")
        return

    log.info(f"Telegram -> token sonu: ...{TELEGRAM_TOKEN[-8:]}  chat_id: {TELEGRAM_CHAT_ID}")

    # Mesajlari olustur ve gonder
    messages = build_telegram_message(top_longs, top_shorts, TIMEFRAME)
    bot = Bot(token=TELEGRAM_TOKEN)

    for i, msg in enumerate(messages):
        try:
            log.info(f"Mesaj {i+1}/{len(messages)} gonderiliyor ({len(msg)} karakter)...")
            sent = await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode=ParseMode.MARKDOWN
            )
            log.info(f"Mesaj {i+1} OK  message_id={sent.message_id}")
            await asyncio.sleep(2)
        except Exception as e:
            log.error(f"Telegram hatasi (mesaj {i+1}): {type(e).__name__}: {e}")
            # Markdown parse hatasi olabilir - plain text ile tekrar dene
            try:
                log.info("Plain text ile tekrar deneniyor...")
                plain = msg.replace("*","").replace("`","").replace("_","")
                sent = await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=plain)
                log.info(f"Plain text OK  message_id={sent.message_id}")
                await asyncio.sleep(2)
            except Exception as e2:
                log.error(f"Plain text de basarisiz: {type(e2).__name__}: {e2}")

    log.info("Tum mesajlar gonderildi [OK]")

# ─── ZAMANLAYICI ────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(run_scan())
