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

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "8402488879:AAHbmCBU2JJS0fsKZyH6xY0SERzkWG-wqWM")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "=1385442139")
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

def calc_fibonacci(high, low, close, period=50):
    """Son N mumun high/low'una gore Fibonacci seviyeleri"""
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
    # Fiyat hangi seviyeye en yakin
    closest = min(levels, key=lambda k: abs(levels[k] - price))
    pct_pos = (price - ll) / diff  # 0=dip, 1=tepe
    if pct_pos < 0.382:
        sig = "long"   # alt bolge - destek
    elif pct_pos > 0.618:
        sig = "short"  # ust bolge - direnc
    else:
        sig = "neutral"
    return sig, closest

def calc_cmf(high, low, close, vol, period=20):
    """Chaikin Money Flow - kurumsal para akisi"""
    mfv = ((close - low) - (high - close)) / (high - low + 1e-10) * vol
    return mfv.rolling(period).sum() / (vol.rolling(period).sum() + 1e-10)

def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend - trend yonu"""
    atr = calc_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    supertrend = close.copy()
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
    """WaveTrend - asiri alim/satim dönüsleri"""
    hlc3 = (high + low + close) / 3
    esa = hlc3.ewm(span=n1, adjust=False).mean()
    d = (hlc3 - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d + 1e-10)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2

def calc_squeeze(high, low, close, vol, bb_period=20, kc_period=20, kc_mult=1.5):
    """Squeeze Momentum - patlama ani tespiti"""
    # Bollinger Bands
    bb_mid = close.rolling(bb_period).mean()
    bb_std = close.rolling(bb_period).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    # Keltner Channels
    kc_atr = calc_atr(high, low, close, kc_period)
    kc_upper = bb_mid + kc_mult * kc_atr
    kc_lower = bb_mid - kc_mult * kc_atr
    # Squeeze: BB icinde KC ise sikisma var
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    # Momentum
    delta = close - (high.rolling(kc_period).max() + low.rolling(kc_period).min()) / 2
    momentum = delta.rolling(bb_period).mean()
    return squeeze.iloc[-1], momentum.iloc[-1], momentum.iloc[-2]

def calc_ichimoku(high, low, close):
    """Ichimoku Bulutu - fiyat bulutun ustunde mi altemi"""
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
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
    if df is None or len(df) < 60: return None
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

    h = calc_macd(close); hv,hp = h.iloc[-1],h.iloc[-2]
    add("MACD", f"{hv:.5f}", "long" if hv>0 and hv>hp else ("short" if hv<0 and hv<hp else "neutral"))

    bbu,bbl = calc_bb(close)
    bb_pct = (price-bbl.iloc[-1])/(bbu.iloc[-1]-bbl.iloc[-1]+1e-10)*100
    add("BB%", f"{bb_pct:.1f}%", "long" if price<bbl.iloc[-1] else ("short" if price>bbu.iloc[-1] else "neutral"))

    e20,e50 = calc_ema(close,20).iloc[-1],calc_ema(close,50).iloc[-1]
    add("EMA20/50", f"{e20/e50:.4f}", "long" if e20>e50 and price>e20 else ("short" if e20<e50 and price<e20 else "neutral"))

    atr_v = calc_atr(high,low,close).iloc[-1]
    ind["ATR%"] = {"value": f"{atr_v/price*100:.2f}%", "signal": "neutral"}

    cci_v = calc_cci(high,low,close).iloc[-1]
    add("CCI(20)", f"{cci_v:.0f}", "long" if cci_v<-100 else ("short" if cci_v>100 else "neutral"))

    wr = calc_wr(high,low,close).iloc[-1]
    add("W%R", f"{wr:.0f}", "long" if wr<-80 else ("short" if wr>-20 else "neutral"))

    # 11. Fibonacci
    fib_sig, fib_lvl = calc_fibonacci(high, low, close)
    add("Fib", f"Lvl:{fib_lvl}", fib_sig)

    # 12. CMF
    cmf_v = calc_cmf(high, low, close, vol).iloc[-1]
    sig = "long" if cmf_v > 0.05 else ("short" if cmf_v < -0.05 else "neutral")
    add("CMF", f"{cmf_v:.3f}", sig)

    # 13. RSI
    rsi_v = calc_rsi(close).iloc[-1]
    sig = "long" if rsi_v < 30 else ("short" if rsi_v > 70 else "neutral")
    add("RSI(14)", f"{rsi_v:.0f}", sig)

    # 14. Supertrend
    st_dir = calc_supertrend(high, low, close)
    sig = "long" if st_dir.iloc[-1] == 1 else "short"
    add("Supertrend", "Buy" if sig=="long" else "Sell", sig)

    # 15. Ichimoku
    ich_sig, ich_pos = calc_ichimoku(high, low, close)
    add("Ichimoku", ich_pos, ich_sig)

    # 16. WaveTrend
    wt1, wt2 = calc_wavetrend(high, low, close)
    wv = wt1.iloc[-1]; wp = wt1.iloc[-2]
    sig = "long" if wv < -60 and wv > wt2.iloc[-1] else ("short" if wv > 60 and wv < wt2.iloc[-1] else "neutral")
    add("WaveTrend", f"{wv:.0f}", sig)

    # 17. Squeeze Momentum
    sq_on, sq_mom, sq_prev = calc_squeeze(high, low, close, vol)
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
    
    # 1. Mesaj: Sadece TOP LONG Sinyalleri
    m1 = hdr + "\n🚀━━━━━ TOP 10 LONG ━━━━━🚀\n" + "\n".join(fmt_block(r,"long") for r in longs)
    
    # 2. Mesaj: Sadece TOP SHORT Sinyalleri (Karakter sınırına takılmaması için temizlendi)
    m2 = hdr + "\n🔻━━━━━ TOP 10 SHORT ━━━━━🔻\n" + "\n".join(fmt_block(r,"short") for r in shorts)
    
    # 3. Mesaj: Tamamen Ayrı Bir Gösterge Rehberi Mesajı
    m3 = (
        "📖 *GÖSTERGE AÇIKLAMALARI REHBERİ*\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "• *MACD:* Trend yönünü ve gücünü ölçer. Pozitif ve bir önceki mumdan yüksekse LONG, negatif ve düşükse SHORT teyididir.\n"
        "• *BB% (Bollinger Bands):* Fiyatın kanalın neresinde olduğunu yüzdeyle ölçer. %0'a yakın veya altındaysa (destek) LONG, %100'e yakın veya üstündeyse (direnç) SHORT sinyalidir.\n"
        "• *EMA20/50:* Kısa/orta vadeli trend. 20'lik ortalama 50'liğin üstündeyse ve fiyat da üstündeyse LONG, tersi durumda SHORT.\n"
        "• *ATR%:* Piyasanın oynaklığını (volatilitesini) yüzdeyle gösterir. TP/SL seviyeleri bu oynaklığa göre otomatik belirlenir.\n"
        "• *CCI(20):* Trend değişimlerini izler. -100'ün altı aşırı satımdır (LONG dönebilir), +100'ün üstü aşırı alımdır (SHORT dönebilir).\n"
        "• *W%R:* Stokastik benzeri hızlı osilatördür. -80'in altı aşırı dip (LONG), -20'nin üstü aşırı tepe (SHORT) bölgesidir.\n"
        "• *Fib (Fibonacci):* Son 50 mumun tepe/dip noktasına göre destek ölçer. Fiyat alt bölgedeyse LONG destek teyidi, üst bölgedeyse SHORT direnç teyididir.\n"
        "• *CMF (Chaikin Money Flow):* Balina/kurumsal para akışını ölçer. 0.05'ten büyükse para girişi (LONG), -0.05'ten küçükse para çıkışı (SHORT) vardır.\n"
        "• *RSI(14):* Güç endeksidir. 30'un altı aşırı satım (LONG yaklaşıyor), 70'in üstü aşırı alım (SHORT yaklaşıyor) demektir.\n"
        "• *Supertrend:* ATR tabanlı trend takipçisidir. 'Buy' verirse yükseliş trendi (LONG), 'Sell' verirse düşüş trendi (SHORT) baskındır.\n"
        "• *Ichimoku:* Fiyatın ana buluta göre konumudur. Bulutun üstündeyse trend güçlüdür (LONG), altındaysa zayıftır (SHORT).\n"
        "• *WaveTrend:* Gelişmiş hacim osilatörüdür. -60'ın altında kesişim yaparsa dip dönüşü (LONG), +60'ın üstünde kesişirse tepe dönüşüdür (SHORT).\n"
        "• *Squeeze:* Patlama ve momentum durumudur. 'Squeezing' yakında sert kırılım geleceğini (sıkışma) bildirir. 'Mom Up' yukarı ivmeyi (LONG), 'Mom Dn' aşağı ivmeyi (SHORT) doğrular."
    )
    return [m1, m2, m3]

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
