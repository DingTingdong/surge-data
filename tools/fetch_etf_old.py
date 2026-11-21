"""0050、00631L 在 2015-06～2019-12 的日線（證交所 STOCK_DAY）→ data/old/etf.csv（研究用，網站不讀）。"""
import csv, os, sys, time, requests
sys.path.insert(0, os.getcwd())
from fetch_etf import H, COLS, num
rows = []
for code in ("0050", "00631L"):
    y, m = 2015, 6
    while (y, m) <= (2019, 12):
        u = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={y}{m:02d}01&stockNo={code}&response=json"
        j = {}
        for k in range(4):
            try: j = requests.get(u, headers=H, timeout=30).json(); break
            except Exception: time.sleep(5 * (k + 1))
        for r in j.get("data") or []:
            ry, rm, rd = r[0].split("/"); d = f"{int(ry) + 1911}-{rm}-{rd}"
            if num(r[6]) is None: continue
            rows.append([d, code] + [str(num(r[i])) for i in (3, 4, 5, 6, 1)])
        print(code, y, m, len(rows), flush=True); time.sleep(3)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
os.makedirs("data/old", exist_ok=True)
with open("data/old/etf.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(COLS); w.writerows(sorted(rows))
print("rows", len(rows))
