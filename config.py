# ============================================================
# CONFIG.PY - Tüm ayarlar burada
# ============================================================

import os

# ===== TELEGRAM =====
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TIMEFRAME        = os.getenv("TIMEFRAME", "Min60")
TOP_RESULTS      = 10
MIN_VOLUME_USDT  = float(os.getenv("MIN_VOLUME_USDT", "1000000"))

# ===== TREND FİLTRESİ =====
TREND_FILTER = {
    "enabled": True,
    "min_aligned": 2,  # Supertrend + Ichimoku + EMA'dan en az 2'si aynı yönde
}

# ===== MİNİMUM SİNYAL SKORU =====
MIN_SCORE = 4.0          # Ağırlıklı skor bu değerin altındaysa listeye girmez
MIN_DIRECTION_SIGNALS = 5  # Long/short için en az 5 net sinyal

# ===== TP/SL AYARLARI =====
TP_SL = {
    "tp1_atr_mult": 1.5,
    "tp2_atr_mult": 3.0,
    "tp3_atr_mult": 5.0,
    "sl_atr_mult": 1.5,
    "min_tp1_percent": 1.0,   # TP1 en az %1
    "min_tp3_percent": 3.0,   # TP3 en az %3
    "max_sl_percent": 3.0,    # SL max %3
}

# ===== İNDİKATÖR AĞIRLIKLARI =====
# Trend çekirdeği 2x, momentum 1.5x, osilatör 1x, yardımcı 0.5x
INDICATOR_WEIGHTS = {
    "Supertrend": 2.0,
    "Ichimoku":   2.0,
    "EMA":        2.0,
    "MACD":       1.5,
    "RSI":        1.5,
    "CMF":        1.5,
    "WaveTrend":  1.0,
    "CCI":        1.0,
    "W%R":        1.0,
    "BB%":        0.5,
    "Fib":        0.5,
    "ATR%":       0.0,   # Skora etki etmez
    "Squeeze":    0.5,
}

# ===== EXCLUDE (kripto olmayan tokenler) =====
# Boş liste = hiçbir token filtrelenmez (senin isteğin)
EXCLUDED_SYMBOLS = []

# ===== ZAMAN DİLİMİ PARAMETRELERİ =====
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

# ===== FİYAT FORMAT =====
def format_price(price):
    """Küçük coinler için dinamik format"""
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.5f}"
    elif price >= 0.0001:
        return f"{price:.6f}"
    else:
        return f"{price:.8f}"
