"""
Futures Scanner - MEXC API - Top 10 Long/Short Sinyali
Zaman dilimine gore otomatik parametre ayari
GitHub Actions ile calisir
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

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "8402488879:AAHbmCBU2JJS0fsKZyH6xY0SERzkWG-wqWM")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1385442139")
TIMEFRAME        = os.getenv("TIMEFRAME", "Min60")
TOP_RESULTS      = 10
MIN_VOLUME_USDT  = float(os.getenv("MIN_VOLUME_USDT", "1000000"))

MEXC_BASE = "https://contract.mexc.com"
HEADERS   = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

TF_PARAMS = {
    "Min15": {
        "min_candles": 40,
        "rsi_p": 7, "macd_fast": 6, "macd_slow": 13, "macd_sig": 4,
        "bb_p": 10, "ema_fast": 10, "ema_slow": 25,
        "atr_p": 7, "cci_p": 10, "wr_p": 7, "fib_p": 20,
        "st_p": 7, "st_mult": 2.0, "sq_kc_mult": 1.0,
        "cmf_p": 10, "wt_n1": 6, "wt_n2": 14,
    },
    "Min60": {
        "min_candles": 60,
        "rsi_p": 14, "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
        "bb_p": 20, "ema_fast": 20, "ema_slow": 50,
        "atr_p": 14, "cci_p": 20, "wr_p": 14, "fib_p": 50,
        "st_p": 10, "st_mult": 3.0, "sq_kc_mult": 1.5,
        "cmf_p": 20, "wt_n1": 10, "wt_n2": 21,
    },
    "Hour4": {
        "min_candles": 60,
        "rsi_p": 14, "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
        "bb_p": 20, "ema_fast": 20, "ema_slow": 50,
        "atr_p": 14, "cci_p": 20, "wr_p": 14, "fib_p": 100,
        "st_p": 10, "st_mult": 3.5, "sq_kc_mult": 2.0,
        "cmf_p": 20, "wt_n1": 10, "wt_n2": 21,
    },
}

def get_tf_params():
    return TF_PARAMS.get(TIMEFRAME, TF_PARAMS["Min60"])

log_formatter  = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler   = logging.FileHandler("scanner.log", encoding="utf-8")
file_handler.setFormatter(log_formatter)
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
log = logging.getLogger(__name__)

def get_symbols():
    r = requests.get(f"{MEXC_BASE}/api/v1/contract/detail", headers=HEADERS, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    symbols = [d["symbol"] for d in data if d.get("settleCoin") == "USDT" and d.get("state") == 0]
    log.info(f"Toplam: {len(data)}, USDT: {len(symbols)}")
    return symbols

def get_klines(symbol, interval="Min60", limit=200):
    url = f"{MEXC_BASE}/api/v1/contract/kline/{symbol}"
    r = requests.get(url, headers=HEADERS, params={"interval": interval, "limit": limit}, timeout=10)
    if r.status_code != 200:
        return None
    raw = r.json().get("data", {})
    if not raw:
        return None
    try:
        df = pd.DataFrame({
            "open":  raw.get("open", []),
            "close": raw.get("close", []),
            "high":  raw.get("high", []),
            "low":   raw.get("low", []),
            "vol":   raw.get("vol", []),
        })
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()
    except:
        return None

def get_ticker(symbol):
    r = requests.get(f"{MEXC_BASE}/api/v1/contract/ticker?symbol={symbol}", headers=HEADERS, timeout=5)
    if r.status_code != 200:
        return None
    return r.json().get("data", None)

def calc_rsi(close, p=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / (l + 1e-10)))

def calc_macd(close, fast=12, slow=26, sig=9):
    ml = close.ewm(span=fast,adjust=False).mean() - close.ewm(span=slow,adjust=False).mean()
    sl = ml.ewm(span=sig,adjust=False).mean()
    return ml - sl

def calc_bb(close, p=20, s=2):
    sma = close.rolling(p).mean()
    sd  = close.rolling(p).std()
    return sma+s*sd, sma-s*sd

def calc_ema(close, p):
    return close.ewm(span=p,adjust=False).mean()

def calc_atr(high, low, close, p=14):
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_cci(high, low, close, p=20):
    tp  = (high+low+close)/3
    sma = tp.rolling(p).mean()
    mad = tp.rolling(p).apply(lambda x: np.abs(x-x.mean()).mean())
    return (tp-sma)/(0.015*mad+1e-10)

def calc_wr(high, low, close, p=14):
    return -100*(high.rolling(p).max()-close)/(high.rolling(p).max()-low.rolling(p).min()+1e-10)

def calc_fibonacci(high, low, close, period=50):
    hh = high.rolling(period).max().iloc[-1]
    ll = low.rolling(period).min().iloc[-1]
    price = close.iloc[-1]
    diff = hh - ll
    if diff == 0:
        return "neutral", "0.5"
    levels = {
        "0.236": hh - 0.236 * diff,
        "0.382": hh - 0.382 * diff,
        "0.500": hh - 0.500 * diff,
        "0.618": hh - 0.618 * diff,
        "0.786": hh - 0.786 * diff,
    }
    closest = min(levels, key=lambda k: abs(levels[k] - price))
    pct_pos = (price - ll) / diff
    sig = "long" if pct_pos < 0.382 else ("short" if pct_pos > 0.618 else "neutral")
    return sig, closest

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

def dot(s):
    return "🟢" if s=="long" else ("🔴" if s=="short" else "🟡")

def analyze(symbol, df, ticker):
    p = get_tf_params()
    if df is None or len(df) < p["min_candles"]:
        return None
    try:
        close,high,low,vol = df["close"],df["high"],df["low"],df["vol"]
        price = close.iloc[-1]
    except Exception as e:
        log.debug(f"{symbol} df hatasi: {e}")
        return None

    ind = {}; scores = []

    def add(name, val, sig):
        ind[name] = {"value": val, "signal": sig}
        scores.append(1 if sig=="long" else (-1 if sig=="short" else 0))

    h = calc_macd(close, p["macd_fast"], p["macd_slow"], p["macd_sig"])
    hv,hp = h.iloc[-1],h.iloc[-2]
    add("MACD", f"{hv:.5f}", "long" if hv>0 and hv>hp else ("short" if hv<0 and hv<hp else "neutral"))

    bbu,bbl = calc_bb(close, p["bb_p"])
    bb_pct = (price-bbl.iloc[-1])/(bbu.iloc[-1]-bbl.iloc[-1]+1e-10)*100
    add("BB%", f"{bb_pct:.1f}%", "long" if price<bbl.iloc[-1] else ("short" if price>bbu.iloc[-1] else "neutral"))

    e_fast = calc_ema(close, p["ema_fast"]).iloc[-1]
    e_slow = calc_ema(close, p["ema_slow"]).iloc[-1]
    add(f"EMA{p['ema_fast']}/{p['ema_slow']}", f"{e_fast/e_slow:.4f}",
        "long" if e_fast>e_slow and price>e_fast else ("short" if e_fast<e_slow and price<e_fast else "neutral"))

    atr_v = calc_atr(high, low, close, p["atr_p"]).iloc[-1]
    ind["ATR%"] = {"value": f"{atr_v/price*100:.2f}%", "signal": "neutral"}

    cci_v = calc_cci(high, low, close, p["cci_p"]).iloc[-1]
    add("CCI", f"{cci_v:.0f}", "long" if cci_v<-100 else ("short" if cci_v>100 else "neutral"))

    wr = calc_wr(high, low, close, p["wr_p"]).iloc[-1]
    add("W%R", f"{wr:.0f}", "long" if wr<-80 else ("short" if wr>-20 else "neutral"))

    fib_sig, fib_lvl = calc_fibonacci(high, low, close, p["fib_p"])
    add("Fib", f"Lvl:{fib_lvl}", fib_sig)

    cmf_v = calc_cmf(high, low, close, vol, p["cmf_p"]).iloc[-1]
    add("CMF", f"{cmf_v:.3f}", "long" if cmf_v>0.05 else ("short" if cmf_v<-0.05 else "neutral"))

    rsi_v = calc_rsi(close, p["rsi_p"]).iloc[-1]
    add(f"RSI({p['rsi_p']})", f"{rsi_v:.0f}", "long" if rsi_v<30 else ("short" if rsi_v>70 else "neutral"))

    st_dir = calc_supertrend(high, low, close, p["st_p"], p["st_mult"])
    sig = "long" if st_dir.iloc[-1]==1 else "short"
    add("Supertrend", "Buy" if sig=="long" else "Sell", sig)

    ich_sig, ich_pos = calc_ichimoku(high, low, close)
    add("Ichimoku", ich_pos, ich_sig)

    wt1, wt2 = calc_wavetrend(high, low, close, p["wt_n1"], p["wt_n2"])
    wv = wt1.iloc[-1]
    sig = "long" if wv<-60 and wv>wt2.iloc[-1] else ("short" if wv>60 and wv<wt2.iloc[-1] else "neutral")
    add("WaveTrend", f"{wv:.0f}", sig)

    sq_on, sq_mom, sq_prev = calc_squeeze(high, low, close, vol, p["bb_p"], p["sq_kc_mult"])
    if sq_on:
        sq_sig, sq_str = "neutral", "Squeezing"
    else:
        sq_sig = "long" if sq_mom>0 and sq_mom>sq_prev else ("short" if sq_mom<0 and sq_mom<sq_prev else "neutral")
        sq_str = "Mom Up" if sq_sig=="long" else ("Mom Dn" if sq_sig=="short" else "Idle")
    add("Squeeze", sq_str, sq_sig)

    try:
        pc  = float(ticker.get("riseFallRate", 0)) * 100
        v24 = float(ticker.get("amount24", ticker.get("volume24", 0)))
        v24h = float(ticker.get("volume24", 0))
        v24h_prev = v24h / (1 + abs(pc) / 100 + 0.001)
        vol_change = ((v24h - v24h_prev) / (v24h_prev + 1)) * 100
    except:
        pc = 0; v24 = 0; vol_change = 0

    return {
        "symbol": symbol.replace("_USDT",""), "price": price,
        "score": sum(scores), "long_count": scores.count(1), "short_count": scores.count(-1),
        "indicators": ind,
        "tp_long": price+2.5*atr_v, "sl_long": price-1.5*atr_v,
        "tp_short": price-2.5*atr_v, "sl_short": price+1.5*atr_v,
        "price_change_24h": pc, "volume_24h": v24, "vol_change_24h": vol_change,
    }

def fmt_vol(v):
    if v>=1e9: return f"${v/1e9:.2f}B"
    if v>=1e6: return f"${v/1e6:.1f}M"
    if v>=1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def fmt_block(r, direction):
    price = r["price"]; pc = r["price_change_24h"]; vc = r.get("vol_change_24h", 0)
    if direction=="long":
        tp,sl,cnt,arrow = r["tp_long"],r["sl_long"],r["long_count"],"🚀 LONG"
    else:
        tp,sl,cnt,arrow = r["tp_short"],r["sl_short"],r["short_count"],"🔻 SHORT"
    tp_pct = (tp-price)/price*100; sl_pct = (sl-price)/price*100
    rr = abs(tp_pct/sl_pct) if sl_pct!=0 else 0
    lines = [f"  {dot(d['signal'])} {n}: {d['value']}" for n,d in r["indicators"].items()]
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"*{r['symbol']}* {arrow} {'🔥'*min(cnt,5)}\n"
        f"💵 `${price:.4f}` | 24h: {'📈' if pc>=0 else '📉'} {pc:+.2f}%\n"
        f"💹 Vol: {fmt_vol(r['volume_24h'])} | {'📈' if vc>=0 else '📉'} {vc:+.1f}%\n"
        f"🎯 TP: `${tp:.4f}` ({tp_pct:+.2f}%)\n"
        f"🛑 SL: `${sl:.4f}` ({sl_pct:+.2f}%)\n"
        f"⚖️ R/R: 1:{rr:.1f}\n"
        f"📈 *Gostergeler*\n" + "\n".join(lines) + "\n"
    )

def build_messages(longs, shorts):
    p   = get_tf_params()
    ts  = datetime.utcnow().strftime("%d.%m.%Y %H:%M")
    hdr = f"🤖 *MEXC Futures* | {ts} UTC | TF: `{TIMEFRAME}`\n🟢 Long  🔴 Short  🟡 Notr\n"
    m1  = hdr + "\n🚀━━━━━ TOP 10 LONG ━━━━━🚀\n" + "\n".join(fmt_block(r,"long")  for r in longs)
    m2  = hdr + "\n🔻━━━━━ TOP 10 SHORT ━━━━━🔻\n" + "\n".join(fmt_block(r,"short") for r in shorts)
    m3  = (
        "📖 *GOSTERGE REHBERI*\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_TF={TIMEFRAME} | RSI={p['rsi_p']} | BB={p['bb_p']} | EMA={p['ema_fast']}/{p['ema_slow']}_\n\n"
        "MACD: Momentum yonu/gucu\n"
        "BB%: Bollinger pozisyonu\n"
        f"EMA{p['ema_fast']}/{p['ema_slow']}: Trend kesisimi\n"
        "ATR%: Volatilite (TP/SL icin)\n"
        "CCI: Trend donusu (-100/+100)\n"
        "W%R: Asiri dip/tepe bolgesi\n"
        "Fib: Fibonacci destek/direnc\n"
        "CMF: Kurumsal para akisi\n"
        f"RSI({p['rsi_p']}): Asiri alim/satim\n"
        "Supertrend: Trend yonu (Buy/Sell)\n"
        "Ichimoku: Bulut ustu/alti/ici\n"
        "WaveTrend: Erken donus sinyali\n"
        "Squeeze: Patlama/momentum"
    )
    return [m1, m2, m3]

async def run_scan():
    log.info(f"MEXC tarama basliyor... TF={TIMEFRAME}")
    try:
        symbols = get_symbols()
        log.info(f"{len(symbols)} sembol bulundu")
    except Exception as e:
        log.error(f"Sembol listesi alinamadi: {e}"); return

    results = []; failed = 0
    for i, sym in enumerate(symbols):
        try:
            df     = get_klines(sym, TIMEFRAME, 200)
            ticker = get_ticker(sym)
            if ticker is None: continue
            if float(ticker.get("amount24", 0)) < MIN_VOLUME_USDT: continue
            result = analyze(sym, df, ticker)
            if result: results.append(result)
            time.sleep(1 if i % 50 == 0 and i > 0 else 0.05)
        except Exception as e:
            failed += 1
            if failed <= 3:
                log.error(f"{sym} HATA: {type(e).__name__}: {e}")
            else:
                log.debug(f"{sym}: {e}")

    log.info(f"Tamamlandi: {len(results)} gecerli, {failed} hata")
    if not results:
        log.warning("Sonuc bulunamadi!"); return

    top_longs  = sorted(results, key=lambda x: x["score"], reverse=True)[:TOP_RESULTS]
    top_shorts = sorted(results, key=lambda x: x["score"])[:TOP_RESULTS]
    log.info(f"En iyi long: {top_longs[0]['symbol']} skor={top_longs[0]['score']}")

    if "YOUR_TOKEN" in TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN ayarlanmamis!"); return

    bot = Bot(token=TELEGRAM_TOKEN)
    for i, msg in enumerate(build_messages(top_longs, top_shorts)):
        try:
            sent = await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
            log.info(f"Mesaj {i+1} OK id={sent.message_id}")
            await asyncio.sleep(2)
        except Exception as e:
            log.error(f"Telegram hatasi mesaj {i+1}: {e}")
            try:
                plain = msg.replace("*","").replace("`","").replace("_","")
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=plain)
            except Exception as e2:
                log.error(f"Plain de basarisiz: {e2}")

    log.info("Tum mesajlar gonderildi [OK]")

if __name__ == "__main__":
    log.info("Scanner basliyor...")
    asyncio.run(run_scan())
