"""
Futures Scanner - OKX API - Top 10 Long/Short Sinyali
Telegram'a TP/SL ve 10 gösterge ile sinyal gonderir
GitHub Actions ile her saat calisir
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
from dotenv import load_dotenv

load_dotenv()

# ─── YAPILANDIRMA ───────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "YOUR_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
TIMEFRAME        = os.getenv("TIMEFRAME", "1H")   # OKX: 1m,5m,15m,1H,4H,1D
TOP_RESULTS      = 10
MIN_VOLUME_USDT  = float(os.getenv("MIN_VOLUME_USDT", "5000000"))

OKX_BASE = "https://www.okx.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# ─── LOGGING ────────────────────────────────────────────────────
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

# ─── OKX API ────────────────────────────────────────────────────
def get_symbols():
    """Tum aktif USDT-margined swap sembolleri al"""
    url = f"{OKX_BASE}/api/v5/public/instruments?instType=SWAP"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]
    symbols = [
        d["instId"] for d in data
        if d["settleCcy"] == "USDT" and d["state"] == "live"
    ]
    return symbols

def get_klines(symbol, bar="1H", limit=200):
    """OHLCV verisi - OKX candles"""
    url = f"{OKX_BASE}/api/v5/market/candles"
    params = {"instId": symbol, "bar": bar, "limit": limit}
    r = requests.get(url, headers=HEADERS, params=params, timeout=10)
    if r.status_code != 200:
        return None
    raw = r.json().get("data", [])
    if not raw:
        return None
    # OKX: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","vol","volCcy","volQuote","confirm"])
    for col in ["open","high","low","close","vol","volQuote"]:
        df[col] = df[col].astype(float)
    df = df.iloc[::-1].reset_index(drop=True)  # eskiden yeniye sirala
    return df

def get_ticker(symbol):
    """24h ticker verisi"""
    url = f"{OKX_BASE}/api/v5/market/ticker?instId={symbol}"
    r = requests.get(url, headers=HEADERS, timeout=5)
    if r.status_code != 200:
        return None
    data = r.json().get("data", [])
    return data[0] if data else None

# ─── TEKNIK GOSTERGELER ─────────────────────────────────────────
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
    return macd_line, signal_line, macd_line - signal_line

def calc_bollinger(close, period=20, std_dev=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma + std_dev*std, sma, sma - std_dev*std

def calc_ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def calc_stochastic(high, low, close, k=14, d=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    stoch_k = 100 * (close - ll) / (hh - ll + 1e-10)
    return stoch_k, stoch_k.rolling(d).mean()

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_obv(close, volume):
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()

def calc_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad + 1e-10)

def calc_williams_r(high, low, close, period=14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll + 1e-10)

def calc_adx(high, low, close, period=14):
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm]   = 0
    minus_dm[minus_dm < plus_dm]  = 0
    atr_val  = calc_atr(high, low, close, period)
    plus_di  = 100 * plus_dm.rolling(period).mean()  / (atr_val + 1e-10)
    minus_di = 100 * minus_dm.rolling(period).mean() / (atr_val + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean(), plus_di, minus_di

# ─── SINYAL NOKTALARI ───────────────────────────────────────────
def dot(sig):
    if sig == "long":  return "🟢"
    if sig == "short": return "🔴"
    return "🟡"

# ─── SEMBOL ANALIZ ──────────────────────────────────────────────
def analyze(symbol, df, ticker):
    if df is None or len(df) < 60:
        return None

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["vol"]
    price = close.iloc[-1]

    ind    = {}
    scores = []

    def add(name, value_str, sig):
        ind[name] = {"value": value_str, "signal": sig}
        scores.append(1 if sig=="long" else (-1 if sig=="short" else 0))

    # 1. RSI
    rsi = calc_rsi(close).iloc[-1]
    sig = "long" if rsi < 35 else ("short" if rsi > 65 else "neutral")
    add("RSI(14)", f"{rsi:.1f}", sig)

    # 2. MACD
    _, _, hist = calc_macd(close)
    h, hp = hist.iloc[-1], hist.iloc[-2]
    sig = "long" if h > 0 and h > hp else ("short" if h < 0 and h < hp else "neutral")
    add("MACD", f"{h:.5f}", sig)

    # 3. Bollinger Band
    bb_up, _, bb_low = calc_bollinger(close)
    bb_pct = (price - bb_low.iloc[-1]) / (bb_up.iloc[-1] - bb_low.iloc[-1] + 1e-10) * 100
    sig = "long" if price < bb_low.iloc[-1] else ("short" if price > bb_up.iloc[-1] else "neutral")
    add("BB%", f"{bb_pct:.1f}%", sig)

    # 4. EMA 20/50
    e20, e50 = calc_ema(close, 20).iloc[-1], calc_ema(close, 50).iloc[-1]
    sig = "long" if e20 > e50 and price > e20 else ("short" if e20 < e50 and price < e20 else "neutral")
    add("EMA20/50", f"{e20/e50:.4f}", sig)

    # 5. Stochastic
    sk, sd = calc_stochastic(high, low, close)
    k_val, d_val = sk.iloc[-1], sd.iloc[-1]
    sig = "long" if k_val < 20 and k_val > d_val else ("short" if k_val > 80 and k_val < d_val else "neutral")
    add("Stoch", f"K:{k_val:.0f} D:{d_val:.0f}", sig)

    # 6. ATR (sadece bilgi, skora katilmaz)
    atr_val = calc_atr(high, low, close).iloc[-1]
    atr_pct = atr_val / price * 100
    ind["ATR%"] = {"value": f"{atr_pct:.2f}%", "signal": "neutral"}

    # 7. OBV
    obv = calc_obv(close, vol)
    obv_ema = obv.ewm(span=20).mean()
    sig = "long" if obv.iloc[-1] > obv_ema.iloc[-1] else ("short" if obv.iloc[-1] < obv_ema.iloc[-1] else "neutral")
    add("OBV", "up" if sig=="long" else ("dn" if sig=="short" else "--"), sig)

    # 8. CCI
    cci_val = calc_cci(high, low, close).iloc[-1]
    sig = "long" if cci_val < -100 else ("short" if cci_val > 100 else "neutral")
    add("CCI(20)", f"{cci_val:.0f}", sig)

    # 9. Williams %R
    wr = calc_williams_r(high, low, close).iloc[-1]
    sig = "long" if wr < -80 else ("short" if wr > -20 else "neutral")
    add("W%R", f"{wr:.0f}", sig)

    # 10. ADX
    adx_val, pdi, mdi = calc_adx(high, low, close)
    av = adx_val.iloc[-1]
    sig = "long" if av > 25 and pdi.iloc[-1] > mdi.iloc[-1] else ("short" if av > 25 and mdi.iloc[-1] > pdi.iloc[-1] else "neutral")
    add("ADX", f"{av:.0f}", sig)

    total = sum(scores)
    lc    = scores.count(1)
    sc    = scores.count(-1)

    # TP/SL
    tp_long  = price + 2.5 * atr_val
    sl_long  = price - 1.5 * atr_val
    tp_short = price - 2.5 * atr_val
    sl_short = price + 1.5 * atr_val

    # 24h degisim
    try:
        price_ch = float(ticker.get("sodUtc8", 0))  # open at day start
        open_px  = float(ticker.get("open24h", price))
        price_ch = (price - open_px) / open_px * 100
        vol_24h  = float(ticker.get("volCcy24h", 0))
    except:
        price_ch = 0
        vol_24h  = 0

    return {
        "symbol": symbol.replace("-USDT-SWAP", ""),
        "instId": symbol,
        "price": price,
        "score": total,
        "long_count": lc,
        "short_count": sc,
        "indicators": ind,
        "tp_long": tp_long, "sl_long": sl_long,
        "tp_short": tp_short, "sl_short": sl_short,
        "price_change_24h": price_ch,
        "volume_24h": vol_24h,
    }

# ─── MESAJ FORMATLAMA ───────────────────────────────────────────
def fmt_vol(v):
    if v >= 1e9: return f"${v/1e9:.2f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def fmt_block(r, direction):
    sym   = r["symbol"]
    price = r["price"]
    ind   = r["indicators"]
    pc    = r["price_change_24h"]
    pc_s  = f"{'up' if pc>=0 else 'dn'} {pc:+.2f}%"
    vol_s = fmt_vol(r["volume_24h"])

    if direction == "long":
        arrow = "LONG"
        tp, sl = r["tp_long"], r["sl_long"]
        cnt = r["long_count"]
    else:
        arrow = "SHORT"
        tp, sl = r["tp_short"], r["sl_short"]
        cnt = r["short_count"]

    tp_pct = (tp - price) / price * 100
    sl_pct = (sl - price) / price * 100
    rr = abs(tp_pct / sl_pct) if sl_pct != 0 else 0
    fire = "🔥" * min(cnt, 5)

    lines = [f"  {dot(d['signal'])} {n}: {d['value']}" for n, d in ind.items()]

    return (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"*{sym}* {'🚀' if direction=='long' else '🔻'} {arrow} {fire}\n"
        f"💵 `${price:.4f}` | 24h: {pc_s}\n"
        f"💹 Vol: {vol_s}\n"
        f"🎯 TP: `${tp:.4f}` ({tp_pct:+.2f}%)\n"
        f"🛑 SL: `${sl:.4f}` ({sl_pct:+.2f}%)\n"
        f"⚖️ R/R: 1:{rr:.1f}\n"
        f"📈 *Gostergeler*\n"
        + "\n".join(lines) + "\n"
    )

def build_messages(longs, shorts):
    ts = datetime.now().strftime("%d.%m.%Y %H:%M")
    hdr = f"🤖 *Futures Tarama* | {ts} | TF: `{TIMEFRAME}`\n🟢 Long  🔴 Short  🟡 Notr\n"
    m1 = hdr + "\n🚀━━━━━ TOP 10 LONG ━━━━━🚀\n" + "\n".join(fmt_block(r,"long")  for r in longs)
    m2 = "\n🔻━━━━━ TOP 10 SHORT ━━━━━🔻\n" + "\n".join(fmt_block(r,"short") for r in shorts)
    m2 += "\n_RSI · MACD · BB · EMA · Stoch · ATR · OBV · CCI · W%R · ADX_"
    return [m1, m2]

# ─── ANA TARAMA ─────────────────────────────────────────────────
async def run_scan():
    log.info("Tarama basliyor... (OKX API)")

    try:
        symbols = get_symbols()
        log.info(f"{len(symbols)} sembol bulundu")
    except Exception as e:
        log.error(f"Sembol listesi alinamadi: {e}")
        return

    results = []
    failed  = 0

    for i, sym in enumerate(symbols):
        try:
            df     = get_klines(sym, TIMEFRAME, 200)
            ticker = get_ticker(sym)

            if ticker is None:
                continue
            vol_24h = float(ticker.get("volCcy24h", 0))
            if vol_24h < MIN_VOLUME_USDT:
                continue

            result = analyze(sym, df, ticker)
            if result:
                results.append(result)

            if i % 50 == 0 and i > 0:
                log.info(f"  {i}/{len(symbols)} tarandi...")
                time.sleep(1)
            else:
                time.sleep(0.05)

        except Exception as e:
            failed += 1
            log.debug(f"{sym}: {e}")

    log.info(f"Tamamlandi: {len(results)} gecerli, {failed} hata")

    if not results:
        log.warning("Sonuc bulunamadi!")
        return

    top_longs  = sorted(results, key=lambda x: x["score"], reverse=True)[:TOP_RESULTS]
    top_shorts = sorted(results, key=lambda x: x["score"])[:TOP_RESULTS]

    log.info(f"En iyi long : {top_longs[0]['symbol']}  skor={top_longs[0]['score']}")
    log.info(f"En iyi short: {top_shorts[0]['symbol']} skor={top_shorts[0]['score']}")

    if "YOUR_TOKEN" in TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN ayarlanmamis!")
        return

    messages = build_messages(top_longs, top_shorts)
    bot = Bot(token=TELEGRAM_TOKEN)

    for i, msg in enumerate(messages):
        try:
            log.info(f"Mesaj {i+1} gonderiliyor...")
            sent = await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode=ParseMode.MARKDOWN
            )
            log.info(f"Mesaj {i+1} OK  id={sent.message_id}")
            await asyncio.sleep(2)
        except Exception as e:
            log.error(f"Telegram hatasi: {e}")
            try:
                plain = msg.replace("*","").replace("`","").replace("_","")
                sent  = await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=plain)
                log.info(f"Plain OK  id={sent.message_id}")
            except Exception as e2:
                log.error(f"Plain de basarisiz: {e2}")

    log.info("Tum mesajlar gonderildi [OK]")

if __name__ == "__main__":
    log.info("Scanner basliyor...")
    asyncio.run(run_scan())
