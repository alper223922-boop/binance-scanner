"""
Futures Scanner - MEXC API - Top 10 Long/Short Sinyali
Telegram'a TP/SL ve 10 gösterge ile sinyal gonderir
Zaman dilimine (15m, 1h, 4h) ozel dinamik parametreler icerir.
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
TIMEFRAME        = os.getenv("TIMEFRAME", "Min60")  # Istenebilir: Min15, Min60, Hour4
TOP_RESULTS      = 10
MIN_VOLUME_USDT  = float(os.getenv("MIN_VOLUME_USDT", "1000000"))

# Zaman dilimlerine gore talep ettiginiz özel konfigurasyon matrisi
TIMEFRAME_CONFIG = {
    "Min15": {
        "rsi_p": 7, "macd_f": 6, "macd_s": 13, "macd_sig": 4, "bb_p": 10,
        "ema_f": 10, "ema_s": 25, "atr_p": 7, "cci_p": 10, "wr_p": 7,
        "fib_p": 20, "st_mult": 2.0, "ich_t": 9, "ich_k": 26, "ich_b": 52,
        "sq_mult": 1.0, "min_len": 40
    },
    "Min60": {
        "rsi_p": 14, "macd_f": 12, "macd_s": 26, "macd_sig": 9, "bb_p": 20,
        "ema_f": 20, "ema_s": 50, "atr_p": 14, "cci_p": 20, "wr_p": 14,
        "fib_p": 50, "st_mult": 3.0, "ich_t": 9, "ich_k": 26, "ich_b": 52,
        "sq_mult": 1.5, "min_len": 60
    },
    "Hour4": {
        "rsi_p": 14, "macd_f": 12, "macd_s": 26, "macd_sig": 9, "bb_p": 20,
        "ema_f": 20, "ema_s": 50, "atr_p": 14, "cci_p": 20, "wr_p": 14,
        "fib_p": 100, "st_mult": 3.5, "ich_t": 9, "ich_k": 26, "ich_b": 52,
        "sq_mult": 2.0, "min_len": 60
    }
}

# Secilen timeframe listede yoksa varsayilan olarak 1 saatlik ayarlar baz alinir
cfg = TIMEFRAME_CONFIG.get(TIMEFRAME, TIMEFRAME_CONFIG["Min60"])

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

def calc_rsi(close, p):
    d = close.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / (l + 1e-10)))

def calc_macd(close, fast, slow, sig):
    ml = close.ewm(span=fast,adjust=False).mean() - close.ewm(span=slow,adjust=False).mean()
    sl = ml.ewm(span=sig,adjust=False).mean()
    return ml - sl

def calc_bb(close, p, s=2):
    sma = close.rolling(p).mean(); sd = close.rolling(p).std()
    return sma+s*sd, sma-s*sd

def calc_ema(close, p):
    return close.ewm(span=p,adjust=False).mean()

def calc_atr(high, low, close, p):
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_cci(high, low, close, p):
    tp = (high+low+close)/3; sma = tp.rolling(p).mean()
    mad = tp.rolling(p).apply(lambda x: np.abs(x-x.mean()).mean())
    return (tp-sma)/(0.015*mad+1e-10)

def calc_wr(high, low, close, p):
    return -100*(high.rolling(p).max()-close)/(high.rolling(p).max()-low.rolling(p).min()+1e-10)

def calc_fibonacci(high, low, close, period):
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
    if pct_pos < 0.382:
        sig = "long"
    elif pct_pos > 0.618:
        sig = "short"
    else:
        sig = "neutral"
    return sig, closest

def calc_cmf(high, low, close, vol, period=20):
    mfv = ((close - low) - (high - close)) / (high - low + 1e-10) * vol
    return mfv.rolling(period).sum() / (vol.rolling(period).sum() + 1e-10)

def calc_supertrend(high, low, close, period, multiplier):
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
    esa = hlc3.ewm(span=n1, adjust=False).mean()
    d = (hlc3 - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d + 1e-10)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2

def calc_squeeze(high, low, close, bb_period, kc_period, kc_mult):
    bb_mid = close.rolling(bb_period).mean()
    bb_std = close.rolling(bb_period).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    kc_atr = calc_atr(high, low, close, kc_period)
    kc_upper = bb_mid + kc_mult * kc_atr
    kc_lower = bb_mid - kc_mult * kc_atr
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    delta = close - (high.rolling(kc_period).max() + low.rolling(kc_period).min()) / 2
    momentum = delta.rolling(bb_period).mean()
    return squeeze.iloc[-1], momentum.iloc[-1], momentum.iloc[-2]

def calc_ichimoku(high, low, close, t, k, b):
    tenkan = (high.rolling(t).max() + low.rolling(t).min()) / 2
    kijun  = (high.rolling(k).max() + low.rolling(k).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(k)
    senkou_b = ((high.rolling(b).max() + low.rolling(b).min()) / 2).shift(k)
    price = close.iloc[-1]
    cloud_top = max(senkou_a.iloc[-1], senkou_b.iloc[-1])
    cloud_bot = min(senkou_a.iloc[-1], senkou_b.iloc[-1])
    if price > cloud_top:
        return "long", "Above"
    elif price < cloud_bot:
        return "short", "Below"
    return "neutral", "Inside"

def dot(s):
    return "🟢" if s=="long" else ("🔴" if s=="short" else "🟡")

def analyze(symbol, df, ticker):
    if df is None or len(df) < cfg["min_len"]: return None
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

    # Dinamik MACD Taraması
    h = calc_macd(close, cfg["macd_f"], cfg["macd_s"], cfg["macd_sig"]); hv, hp = h.iloc[-1], h.iloc[-2]
    add("MACD", f"{hv:.5f}", "long" if hv>0 and hv>hp else ("short" if hv<0 and hv<hp else "neutral"))

    # Dinamik Bollinger Taraması
    bbu,bbl = calc_bb(close, cfg["bb_p"])
    bb_pct = (price-bbl.iloc[-1])/(bbu.iloc[-1]-bbl.iloc[-1]+1e-10)*100
    add("BB%", f"{bb_pct:.1f}%", "long" if price<bbl.iloc[-1] else ("short" if price>bbu.iloc[-1] else "neutral"))

    # Dinamik EMA Taraması
    e20,e50 = calc_ema(close, cfg["ema_f"]).iloc[-1], calc_ema(close, cfg["ema_s"]).iloc[-1]
    add(f"EMA{cfg['ema_f']}/{cfg['ema_s']}", f"{e20/e50:.4f}", "long" if e20>e50 and price>e20 else ("short" if e20<e50 and price<e20 else "neutral"))

    # Dinamik ATR ve Risk Hesabı
    atr_v = calc_atr(high,low,close, cfg["atr_p"]).iloc[-1]
    ind["ATR%"] = {"value": f"{atr_v/price*100:.2f}%", "signal": "neutral"}

    # Dinamik CCI Taraması
    cci_v = calc_cci(high,low,close, cfg["cci_p"]).iloc[-1]
    add("CCI", f"{cci_v:.0f}", "long" if cci_v<-100 else ("short" if cci_v>100 else "neutral"))

    # Dinamik Williams %R Taraması
    wr = calc_wr(high,low,close, cfg["wr_p"]).iloc[-1]
    add("W%R", f"{wr:.0f}", "long" if wr<-80 else ("short" if wr>-20 else "neutral"))

    # Dinamik Fibonacci Taraması
    fib_sig, fib_lvl = calc_fibonacci(high, low, close, cfg["fib_p"])
    add("Fib", f"Lvl:{fib_lvl}", fib_sig)

    # Chaikin Money Flow (Sabit 20 periyot)
    cmf_v = calc_cmf(high, low, close, vol, period=20)
    sig = "long" if cmf_v > 0.05 else ("short" if cmf_v < -0.05 else "neutral")
    add("CMF", f"{cmf_v:.3f}", sig)

    # Dinamik RSI Taraması
    rsi_v = calc_rsi(close, cfg["rsi_p"]).iloc[-1]
    sig = "long" if rsi_v < 30 else ("short" if rsi_v > 70 else "neutral")
    add(f"RSI({cfg['rsi_p']})", f"{rsi_v:.0f}", sig)

    # Dinamik Supertrend Taraması
    st_dir = calc_supertrend(high, low, close, cfg["atr_p"], cfg["st_mult"])
    sig = "long" if st_dir.iloc[-1] == 1 else "short"
    add("Supertrend", "Buy" if sig=="long" else "Sell", sig)

    # Dinamik Ichimoku Taraması
    ich_sig, ich_pos = calc_ichimoku(high, low, close, cfg["ich_t"], cfg["ich_k"], cfg["ich_b"])
    add("Ichimoku", ich_pos, ich_sig)

    # WaveTrend (Sabit 10/21)
    wt1, wt2 = calc_wavetrend(high, low, close)
    wv = wt1.iloc[-1]
    sig = "long" if wv < -60 and wv > wt2.iloc[-1] else ("short" if wv > 60 and wv < wt2.iloc[-1] else "neutral")
    add("WaveTrend", f"{wv:.0f}", sig)

    # Dinamik Squeeze Momentum Taraması
    sq_on, sq_mom, sq_prev = calc_squeeze(high, low, close, cfg["bb_p"], cfg["bb_p"], cfg["sq_mult"])
    if sq_on:
        sq_sig = "neutral"
        sq_str = "Squeezing"
    else:
        sq_sig = "long" if sq_mom > 0 and sq_mom > sq_prev else ("short" if sq_mom < 0 and sq_mom < sq_prev else "neutral")
        sq_str = "Mom Up" if sq_sig=="long" else ("Mom Dn" if sq_sig=="short" else "Idle")
    add("Squeeze", sq_str, sq_sig)

    try:
        pc   = float(ticker.get("riseFallRate", 0)) * 100
        v24  = float(ticker.get("amount24", ticker.get("volume24", 0)))
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
    m2 = hdr + "\n🔻━━━━━ TOP 10 SHORT ━━━━━🔻\n" + "\n".join(fmt_block(r,"short") for r in shorts)
    
    m3 = (
        f"📖 *GÖSTERGE REHBERİ ({TIMEFRAME} ÖZEL)*\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "• *MACD:* Trend yonunu olcer. Hizli donus teyididir.\n"
        "• *BB%:* Bollinger kanal yerlesimidir. Kanalin disi asiri uc bolgedir.\n"
        f"• *EMA {cfg['ema_f']}/{cfg['ema_s']}:* Dinamik trend yonu. Fiyat ve kisa vade ustteyse LONG, alttayda SHORT.\n"
        "• *ATR%:* Piyasa oynakligi. TP/SL limitlerini belirler.\n"
        "• *CCI / W%R:* Hizli asiri alim/satim osilatörleridir.\n"
        f"• *Fib:* Son {cfg['fib_p']} mumun tepesine gore destek ve direnc bulur.\n"
        "• *CMF:* Kurumsal/Balina para akisidir. Pozitifse LONG, negatifse SHORT teyit eder.\n"
        f"• *RSI({cfg['rsi_p']}):* Guc endeksidir. Zaman dilimine ozel periyot kullanir.\n"
        f"• *Supertrend (Mult:{cfg['st_mult']}):* Ana trend yonunu keskin sekilde takip eder.\n"
        "• *Ichimoku:* Fiyat bulut iliskisidir. Ustü guclü trend, alti zayif trenddir.\n"
        "• *WaveTrend:* Gelismis dip/tepe donus sinyalidir.\n"
        f"• *Squeeze (Mult:{cfg['sq_mult']}):* 'Squeezing' daralma-patlama habercisidir. 'Mom' etiketleri yonu dogrular."
    )
    return [m1, m2, m3]

async def run_scan():
    log.info(f"MEXC tarama basliyor... Zaman Dilimi: {TIMEFRAME}")
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
            failed += 1
            if failed <= 5:
                log.error(f"{sym} HATA: {type(e).__name__}: {e}")
            else:
                log.debug(f"{sym}: {e}")

    log.info(f"Tamamlandi: {len(results)} gecerli, {failed} hata")
    if not results:
        log.warning("Sonuc bulunamadi!"); return

    top_longs  = sorted(results, key=lambda x: x["score"], reverse=True)[:TOP_RESULTS]
    top_shorts = sorted(results, key=lambda x: x["score"])[:TOP_RESULTS]

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
