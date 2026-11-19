"""抓上市櫃公司月營收（公開資訊觀測站每月彙總表），存到 data/revenue/YYYY.csv.gz（往年）與 data/revenue/YYYY.csv（今年）。

欄位：ym（營收月份 YYYY-MM）, code, rev（當月營收，千元）, mom（比上月 %）, yoy（比去年同月 %）, cum_yoy（累計比去年 %）
法規：每月 10 日前公布上個月營收。研究時把 YYYY-MM 的營收當成「下個月 11 日起」才知道。

用法：python fetch_revenue.py              → 重抓最近 3 個月（每天跑，月初各家陸續公布）
      python fetch_revenue.py 2019-01 2025-12 → 補指定月份（歷史）
"""
import csv, gzip, os, re, sys, time, datetime as dt, requests

H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}
COLS = ["ym", "code", "rev", "mom", "yoy", "cum_yoy"]

def num(x):
    x = x.replace(",", "").strip()
    try: return float(x)
    except ValueError: return None

def month_rows(y, m):
    out, ok = {}, False
    for mk in ("sii", "otc"):
        for kind in (0, 1):  # 0 國內公司、1 外國公司（KY）
            u = f"https://mopsov.twse.com.tw/nas/t21/{mk}/t21sc03_{y - 1911}_{m}_{kind}.html"
            html = None
            for i in range(4):
                try:
                    r = requests.get(u, headers=H, timeout=60)
                    if r.status_code == 200: html = r.content.decode("utf-8", "replace") if b"charset=utf-8" in r.content[:2000].lower() else r.content.decode("big5", "replace"); break
                    if r.status_code == 404: break
                except Exception: pass
                time.sleep(5 * (i + 1))
            time.sleep(2)
            if html is None: continue
            ok = True
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
                c = [re.sub(r"<[^>]+>|&nbsp;", "", x).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S | re.I)]
                if len(c) >= 10 and re.fullmatch(r"[1-9]\d{3}", c[0]):
                    out[c[0]] = [f"{y}-{m:02d}", c[0], num(c[2]), num(c[5]), num(c[6]), num(c[9])]
    return list(out.values()), ok

def opener(fn, mode):
    if fn.endswith(".gz"): return gzip.open(fn, mode + "t", encoding="utf-8", newline="" if mode == "w" else None)
    return open(fn, mode, encoding="utf-8", newline="" if mode == "w" else None)

def read(fn):
    if not os.path.exists(fn): return []
    with opener(fn, "r") as f: return list(csv.reader(f))[1:]

def write(fn, rows):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with opener(fn, "w") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(sorted(rows, key=lambda r: (r[0], r[1])))

def months(a, b):
    y, m = a
    while (y, m) <= b:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

def main(start=None, end=None):
    today = dt.date.today()
    if start:
        a = tuple(map(int, start.split("-"))); b = tuple(map(int, end.split("-")))
    else:
        b = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        a = b
        for _ in range(2): a = (a[0], a[1] - 1) if a[1] > 1 else (a[0] - 1, 12)
    by_year = {}
    for y, m in months(a, b):
        rows, ok = month_rows(y, m)
        print(f"{y}-{m:02d}", len(rows), "rows" if ok else "FAILED", flush=True)
        if ok and rows: by_year.setdefault(y, {})[f"{y}-{m:02d}"] = rows
    for y, got in by_year.items():
        fn = f"data/revenue/{y}.csv" if y == today.year else f"data/revenue/{y}.csv.gz"
        old = [r for r in read(fn) if r[0] not in got]
        new = old + [[str(v) if v is not None else "" for v in r] for rs in got.values() for r in rs]
        if sorted(new) != sorted(read(fn)): write(fn, new); print("wrote", fn, len(new))
        else: print("unchanged", fn)

if __name__ == "__main__":
    main(*sys.argv[1:3])
