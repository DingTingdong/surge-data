"""把 fetch_history.py 抓回來的各年檔案併進 repo。

- 早於目前年度的年份 → data/history/YYYY.csv.gz
- 目前年度（prices.csv 所在年度）→ 併進 data/prices.csv；同一天同一市場以新抓的為準（有 chg 欄）
- 兩個市場都空的日子 → 加進 data/holidays.txt；抓取失敗的日子 → data/history/failed.txt
"""
import glob, os, sys, pandas as pd
from fetch_prices import COLS

src = sys.argv[1]
os.makedirs("data/history", exist_ok=True)
cur = pd.read_csv("data/prices.csv", dtype={"code": str}) if os.path.exists("data/prices.csv") else pd.DataFrame(columns=COLS)
for c in COLS:
    if c not in cur.columns: cur[c] = None
cur_year = str(cur.date.max())[:4] if len(cur) else "9999"

for fn in sorted(glob.glob(os.path.join(src, "*.csv.gz"))):
    y = os.path.basename(fn)[:4]
    h = pd.read_csv(fn, dtype={"code": str})
    if not len(h): continue
    if y < cur_year:
        out = f"data/history/{y}.csv.gz"
        if os.path.exists(out):
            old = pd.read_csv(out, dtype={"code": str})
            keys = set(zip(h.date, h.market))
            h = pd.concat([old[[k not in keys for k in zip(old.date, old.market)]], h])
        h = h.drop_duplicates(["date", "market", "code"], keep="last").sort_values(["date", "market", "code"])
        h[COLS].to_csv(out, index=False, compression="gzip")
        print(y, "->", out, len(h), "rows")
    else:
        keys = set(zip(h.date, h.market))
        cur = pd.concat([cur[[k not in keys for k in zip(cur.date, cur.market)]], h])
        cur = cur.drop_duplicates(["date", "market", "code"], keep="last").sort_values(["date", "market", "code"])
        print(y, "-> data/prices.csv", len(h), "rows merged,", len(cur), "total")
cur[COLS].to_csv("data/prices.csv", index=False)

hol = set(open("data/holidays.txt").read().split()) if os.path.exists("data/holidays.txt") else set()
for fn in glob.glob(os.path.join(src, "*.closed.txt")):
    hol |= set(open(fn).read().split())
open("data/holidays.txt", "w").write("\n".join(sorted(hol)))
fails = set()
for fn in glob.glob(os.path.join(src, "*.failed.txt")):
    fails |= set(open(fn).read().split())
if fails:
    with open("data/history/failed.txt", "a") as f: f.write("\n".join(sorted(fails)) + "\n")
    print("failed days:", len(fails))
