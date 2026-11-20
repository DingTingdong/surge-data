"""試抓證交所即時報價（mis.twse.com.tw），看雲端主機能不能用、一次能查幾檔、多久更新。"""
import json, time, requests
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://mis.twse.com.tw/stock/index.jsp"}
s = requests.Session()
try: s.get("https://mis.twse.com.tw/stock/index.jsp", headers=H, timeout=20)
except Exception as e: print("index ERR", e)
codes = ["tse_2330.tw", "tse_2468.tw", "tse_2471.tw", "otc_6925.tw", "tse_7765.tw", "otc_7828.tw"]
for n in (6, 50, 150):
    ex = "|".join((codes * 30)[:n])
    u = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={ex}&json=1&delay=0&_={int(time.time()*1000)}"
    t0 = time.time()
    try:
        r = s.get(u, headers=H, timeout=20); j = r.json()
        arr = j.get("msgArray", [])
        print(n, "status", r.status_code, "rows", len(arr), "sec %.1f" % (time.time() - t0))
        for a in arr[:3]: print({k: a.get(k) for k in ("c", "n", "z", "y", "o", "h", "l", "v", "t", "d", "u")})
    except Exception as e: print(n, "ERR", e)
    time.sleep(2)
