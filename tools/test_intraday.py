"""測試盤中收集的掃描（不等開盤、不看日期、只掃一輪、不存檔）。"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import intraday_collect as ic, requests
s = requests.Session(); s.get("https://mis.twse.com.tw/stock/index.jsp", headers=ic.H, timeout=20)
cs = ic.codes(); t0 = time.time(); got = 0; hits = 0; nz = 0
for i in range(0, len(cs), 150):
    arr = ic.fetch(s, cs[i:i + 150]); got += len(arr)
    for a in arr:
        y, z, u = ic.num(a.get("y")), ic.num(a.get("z")), ic.num(a.get("u"))
        if z is None: nz += 1; continue
        if y and z >= y * 1.05: hits += 1
    time.sleep(0.5)
print("codes", len(cs), "rows", got, "no-price", nz, "≥+5%", hits, "seconds %.0f" % (time.time() - t0))
