"""集保戶股權分散表（每週）：只保留當週資料，所以每次更新都檢查、沒存過的週就存下來，累積成歷史（研究用，網站不讀）。
來源：集保結算所開放資料 https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5（每週最後一個營業日收盤後更新）。
存成 data/tdcc/YYYYMMDD.csv.gz（資料日期），只保留四碼普通股。
"""
import io, os, socket, sys, time, requests, pandas as pd
import urllib3.util.connection as _uc
_uc.allowed_gai_family = lambda: socket.AF_INET   # 雲端主機走 IPv6 會「Network is unreachable」，強制 IPv4
URLS = ["https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5", "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"]
os.makedirs("data/tdcc", exist_ok=True)
r = None
for i in range(6):
    u = URLS[i % 2]
    try:
        r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        if r.status_code == 200 and len(r.content) > 100000: print("ok", u); break
        print("bad", u, r.status_code, len(r.content))
    except Exception as e: print("retry", u, e)
    r = None; time.sleep(10)
if r is None:
    print("tdcc fetch failed"); sys.exit(0)
d = pd.read_csv(io.BytesIO(r.content), dtype=str, encoding="utf-8-sig")
d.columns = [c.strip() for c in d.columns]
dc = [c for c in d.columns if "日期" in c][0]; cc = [c for c in d.columns if "代號" in c][0]
d = d[d[cc].str.fullmatch(r"[1-9]\d{3}")]
day = d[dc].iloc[0].strip(); fn = f"data/tdcc/{day}.csv.gz"
if os.path.exists(fn): print("already have", day)
else: d.to_csv(fn, index=False, compression="gzip"); print("saved", fn, len(d), "rows")
