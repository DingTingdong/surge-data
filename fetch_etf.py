"""抓 ETF 日線（證交所 STOCK_DAY，每檔每月一次）→ data/etf.csv。給追蹤頁比較用：0050、00631L（正2）、00632R（反1）。

用法：python fetch_etf.py            → 只重抓這個月和上個月（每天跑）
      python fetch_etf.py full       → 從 2019/10 全部重抓
"""
import csv, os, sys, time, datetime as dt, requests
H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}
CODES = ["0050", "00631L", "00632R"]
OUT = "data/etf.csv"
COLS = ["date", "code", "open", "high", "low", "close", "volume"]
def num(x):
    try: return float(str(x).replace(",", ""))
    except ValueError: return None
def months(full):
    t = dt.date.today()
    if full: y, m = 2019, 10
    else: y, m = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
    while (y, m) <= (t.year, t.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
def main(full=False):
    rows = {}
    if os.path.exists(OUT) and not full:
        with open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f): rows[(r["date"], r["code"])] = [r[c] for c in COLS]
    before = dict(rows)
    for code in CODES:
        for y, m in months(full):
            u = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={y}{m:02d}01&stockNo={code}&response=json"
            j = {}
            for k in range(4):
                try: j = requests.get(u, headers=H, timeout=30).json(); break
                except Exception: time.sleep(5 * (k + 1))
            for r in j.get("data") or []:
                ry, rm, rd = r[0].split("/"); d = f"{int(ry) + 1911}-{rm}-{rd}"
                if num(r[6]) is None: continue
                rows[(d, code)] = [d, code] + [str(num(r[i])) for i in (3, 4, 5, 6, 1)]
            time.sleep(3)
    if rows != before or full:
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(COLS); w.writerows(sorted(rows.values()))
        print("wrote", len(rows))
    else: print("unchanged", len(rows))
if __name__ == "__main__":
    main(len(sys.argv) > 1 and sys.argv[1] == "full")
