"""抓 ETF 日線（證交所 STOCK_DAY，每個月一次）→ data/etf.csv。給研究用：0050、00631L（正2）、00632R（反1）。"""
import csv, os, time, datetime as dt, requests
H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}
CODES = ["0050", "00631L", "00632R"]
OUT = "data/etf.csv"
def num(x):
    try: return float(str(x).replace(",", ""))
    except ValueError: return None
rows = {}
today = dt.date.today()
for code in CODES:
    y, m = 2019, 10
    while (y, m) <= (today.year, today.month):
        u = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={y}{m:02d}01&stockNo={code}&response=json"
        for k in range(4):
            try:
                j = requests.get(u, headers=H, timeout=30).json(); break
            except Exception: time.sleep(5 * (k + 1)); j = {}
        for r in j.get("data") or []:
            ry, rm, rd = r[0].split("/"); d = f"{int(ry) + 1911}-{rm}-{rd}"
            rows[(d, code)] = [d, code, num(r[3]), num(r[4]), num(r[5]), num(r[6]), num(r[1])]
        print(code, y, m, len(j.get("data") or []), flush=True)
        time.sleep(3)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["date", "code", "open", "high", "low", "close", "volume"]); w.writerows(sorted(rows.values()))
print("wrote", len(rows))
