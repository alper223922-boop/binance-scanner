"""
Futures Scanner - MEXC API - Top 10 Long/Short Sinyali
Telegram'a TP/SL ve 10 gösterge ile sinyal gonderir
GitHub Actions ile her 30 dakikada calisir
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

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "YOUR_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
TIMEFRAME        = os.getenv("TIMEFRAME", "Min60")
TOP_RESULTS      = 10
MIN_VOLUME_USDT  = float(os.getenv("MIN_VOLUME_USDT", "1000000"))

MEXC_BASE = "https://contract.mexc.com"
HEADERS   = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

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
    if not symbols:
        log.warning(f"Ornek: {data[:2] if data else bos}")
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
    sma = close.rolling(p).mean(); sd = close.rolling(p).std()
    return sma+s*sd, sma-s*sd

def calc_ema(close, p):
    return close.ewm(span=p,adjust=False).mean()

def calc_stoch(high, low, close, k=14, d=3):
    sk = 100*(close-low.rolling(k).min())/(high.rolling(k).max()-low.rolling(k).min()+1e-10)
    return sk, sk.rolling(d).mean()

def calc_atr(high, low, close, p=14):
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_obv(close, vol):
    return (np.sign(close.diff()).fillna(0)*vol).cumsum()

def calc_cci(high, low, close, p=20):
    tp = (high+low+close)/3; sma = tp.rolling(p).mean()
    mad = tp.rolling(p).apply(lambda x: np.abs(x-x.mean()).mean())
    return (tp-sma)/(0.015*mad+1e-10)

def calc_wr(high, low, close, p=14):
    return -100*(high.rolling(p).max()-close)/(high.rolling(p).max()-low.rolling(p).min()+1e-10)

def calc_adx(high, low, close, p=14):
    pdm = high.diff().clip(lower=0); mdm = (-low.diff()).clip(lower=0)
    pdm[pdm<mdm]=0; mdm[mdm<pdm]=0
    atr = calc_atr(high,low,close,p)
    pdi = 100*pdm.rolling(p).mean()/(atr+1e-10)
    mdi = 100*mdm.rolling(p).mean()/(atr+1e-10)
    dx  = 100*(pdi-mdi).abs()/(pdi+mdi+1e-10)
    return dx.rolling(p).mean(), pdi, mdi

def dot(s):
    return "🟢" if s=="long" else ("🔴" if s=="short" else "🟡")

def analyze(symbol, df, ticker):
    if df is None or len(df) < 60: return None
    close,high,low,vol = df["close"],df["high"],df["low"],df["vol"]
    price = close.iloc[-1]
    ind = {}; scores = []

    def add(name, val, sig):
        ind[name] = {"value": val, "signal": sig}
        scores.append(1 if sig=="long" else (-1 if sig=="short" else 0))

    rsi = calc_rsi(close).iloc[-1]
    add("RSI(14)", f"{rsi:.1f}", "long" if rsi<35 else ("short" if rsi>65 else "neutral"))

    h = calc_macd(close); hv,hp = h.iloc[-1],h.iloc[-2]
    add("MACD", f"{hv:.5f}", "long" if hv>0 and hv>hp else ("short" if hv<0 and hv<hp else "neutral"))

    bbu,bbl = calc_bb(close)
    bb_pct = (price-bbl.iloc[-1])/(bbu.iloc[-1]-bbl.iloc[-1]+1e-10)*100
    add("BB%", f"{bb_pct:.1f}%", "long" if price<bbl.iloc[-1] else ("short" if price>bbu.iloc[-1] else "neutral"))

    e20,e50 = calc_ema(close,20).iloc[-1],calc_ema(close,50).iloc[-1]
    add("EMA20/50", f"{e20/e50:.4f}", "long" if e20>e50 and price>e20 else ("short" if e20<e50 and price<e20 else "neutral"))

    sk,sd = calc_stoch(high,low,close); kv,dv = sk.iloc[-1],sd.iloc[-1]
    add("Stoch", f"K:{kv:.0f} D:{dv:.0f}", "long" if kv<20 and kv>dv else ("short" if kv>80 and kv<dv else "neutral"))

    atr_v = calc_atr(high,low,close).iloc[-1]
    ind["ATR%"] = {"value": f"{atr_v/price*100:.2f}%", "signal": "neutral"}

    obv = calc_obv(close,vol); oe = obv.ewm(span=20).mean()
    add("OBV","up" if obv.iloc[-1]>oe.iloc[-1] else "dn","long" if obv.iloc[-1]>oe.iloc[-1] else ("short" if obv.iloc[-1]<oe.iloc[-1] else "neutral"))

    cci_v = calc_cci(high,low,close).iloc[-1]
    add("CCI(20)", f"{cci_v:.0f}", "long" if cci_v<-100 else ("short" if cci_v>100 else "neutral"))

    wr = calc_wr(high,low,close).iloc[-1]
    add("W%R", f"{wr:.0f}", "long" if wr<-80 else ("short" if wr>-20 else "neutral"))

    adx_v,pdi,mdi = calc_adx(high,low,close); av = adx_v.iloc[-1]
    add("ADX", f"{av:.0f}", "long" if av>25 and pdi.iloc[-1]>mdi.iloc[-1] else ("short" if av>25 and mdi.iloc[-1]>pdi.iloc[-1] else "neutral"))

    try:
        # MEXC ticker alanlari: lastPrice, riseFallRate, volume24, amount24
        pc  = float(ticker.get("riseFallRate", 0)) * 100  # zaten yuzde olarak geliyor
        v24 = float(ticker.get("amount24", ticker.get("volume24", 0)))
    except:
        pc = 0; v24 = 0

    return {
        "symbol": symbol.replace("_USDT",""), "price": price,
        "score": sum(scores), "long_count": scores.count(1), "short_count": scores.count(-1),
        "indicators": ind,
        "tp_long": price+2.5*atr_v, "sl_long": price-1.5*atr_v,
        "tp_short": price-2.5*atr_v, "sl_short": price+1.5*atr_v,
        "price_change_24h": pc, "volume_24h": v24,
    }

def fmt_vol(v):
    if v>=1e9: return f"${v/1e9:.2f}B"
    if v>=1e6: return f"${v/1e6:.1f}M"
    if v>=1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def fmt_block(r, direction):
    price = r["price"]; pc = r["price_change_24h"]
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
        f"💵 `${price:.4f}` | 24h: {'up' if pc>=0 else 'dn'} {pc:+.2f}%\n"
        f"💹 Vol: {fmt_vol(r['volume_24h'])}\n"
        f"🎯 TP: `${tp:.4f}` ({tp_pct:+.2f}%)\n"
        f"🛑 SL: `${sl:.4f}` ({sl_pct:+.2f}%)\n"
        f"⚖️ R/R: 1:{rr:.1f}\n"
        f"📈 *Gostergeler*\n" + "\n".join(lines) + "\n"
    )

def build_messages(longs, shorts):
    ts = datetime.utcnow().strftime("%d.%m.%Y %H:%M")
    hdr = f"🤖 *MEXC Futures Tarama* | {ts} UTC | TF: `{TIMEFRAME}`\n🟢 Long  🔴 Short  🟡 Notr\n"
    m1 = hdr + "\n🚀━━━━━ TOP 10 LONG ━━━━━🚀\n" + "\n".join(fmt_block(r,"long") for r in longs)
    m2 = "\n🔻━━━━━ TOP 10 SHORT ━━━━━🔻\n" + "\n".join(fmt_block(r,"short") for r in shorts)
    m2 += "\n_RSI · MACD · BB · EMA · Stoch · ATR · OBV · CCI · W%R · ADX_"
    return [m1, m2]

async def run_scan():
    log.info("MEXC tarama basliyor...")
    try:
        symbols = get_symbols()
        log.info(f"{len(symbols)} sembol bulundu")
    except Exception as e:
        log.error(f"Sembol listesi alinamadi: {e}"); return

    results = []; failed = 0
    for i, sym in enumerate(symbols):
        try:
            df = get_klines(sym, TIMEFRAME, 200)
            ticker = get_ticker(sym)
            if ticker is None: continue
            if float(ticker.get("amount24", 0)) < MIN_VOLUME_USDT: continue
            result = analyze(sym, df, ticker)
            if result: results.append(result)
            time.sleep(1 if i % 50 == 0 and i > 0 else 0.05)
        except Exception as e:
            failed += 1; log.debug(f"{sym}: {e}")

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
            log.error(f"Telegram hatasi: {e}")
            try:
                plain = msg.replace("*","").replace("`","").replace("_","")
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=plain)
            except Exception as e2:
                log.error(f"Plain de basarisiz: {e2}")

    log.info("Tum mesajlar gonderildi [OK]")

if __name__ == "__main__":
    log.info("Scanner basliyor...")
    asyncio.run(run_scan())
