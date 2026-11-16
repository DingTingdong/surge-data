"""移除櫃買中心在休市日回傳的假資料（舊網址遇到休市會回傳最新一天的行情）。
規則：證交所當天沒有任何資料的日期，櫃買的列一律刪除，並把該日期記成休市。"""
import glob, os, pandas as pd
files = sorted(glob.glob("data/history/*.csv.gz")) + ["data/prices.csv"]
hol = set(open("data/holidays.txt").read().split()) if os.path.exists("data/holidays.txt") else set()
for fn in files:
    d = pd.read_csv(fn, dtype={"code": str})
    tw = set(d[d.market == "TWSE"].date)
    bad = (d.market == "TPEX") & ~d.date.isin(tw)
    dates = sorted(set(d[bad].date))
    hol |= set(dates)
    d = d[~bad]
    if fn.endswith(".gz"): d.to_csv(fn, index=False, compression="gzip")
    else: d.to_csv(fn, index=False)
    print(fn, "removed", int(bad.sum()), "rows on", len(dates), "dates")
open("data/holidays.txt", "w").write("\n".join(sorted(hol)))
