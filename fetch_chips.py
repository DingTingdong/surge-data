"""抓籌碼資料：三大法人買賣超、融資融券餘額（上市＋上櫃，四碼個股），存到 data/chips/YYYY.csv.gz。

欄位（張＝1,000股）：date, market, code, foreign（外資及陸資合計買賣超，含外資自營商）, trust（投信）,
dealer（自營商合計）, inst（三大法人合計）, margin（融資餘額）, short（融券餘額）
抓不到的欄位留空。

用法：python fetch_chips.py                       → 補今年還沒抓的交易日（交易日以行情資料為準）
      python fetch_chips.py 2020-01-01 2020-12-31 → 補指定期間（給 chips-history 工作流程用）
"""
import csv, glob, gzip, os, re, sys, time, datetime as dt, requests

COLS = ["date", "market", "code", "foreign", "trust", "dealer", "inst", "margin", "short"]
H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}

def get(url):
    for i in range(4):
        try:
            r = requests.get(url, headers=H, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(4 * (i + 1))
    return None

def num(x):
    try: return float(str(x).replace(",", "").strip())
    except ValueError: return None

def lots(x):
    v = num(x); return None if v is None else round(v / 1000, 3)

def four(c):
    c = str(c).strip(); return c if re.fullmatch(r"[1-9]\d{3}", c) else None

def twse_inst(d):
    j = get(f"https://www.twse.com.tw/rwd/zh/fund/T86?date={d:%Y%m%d}&selectType=ALLBUT0999&response=json")
    if j is None: return None
    f = j.get("fields") or []
    def k(name):
        for i, x in enumerate(f):
            if x.replace(" ", "") == name: return i
    kc, kf1, kf2, kt, kd, ka = (k("證券代號"), k("外陸資買賣超股數(不含外資自營商)"), k("外資自營商買賣超股數"),
                                k("投信買賣超股數"), k("自營商買賣超股數"), k("三大法人買賣超股數"))
    if None in (kc, kt, ka): kc, kf1, kf2, kt, kd, ka = 0, 4, 7, 10, 11, 18  # 已確認的欄位位置
    out = {}
    for r in j.get("data") or []:
        c = four(r[kc])
        if not c: continue
        fo = (num(r[kf1]) or 0) + (num(r[kf2]) or 0) if kf1 is not None else None
        out[c] = {"foreign": None if fo is None else round(fo / 1000, 3), "trust": lots(r[kt]), "dealer": lots(r[kd]), "inst": lots(r[ka])}
    return out

def tpex_inst(d):
    j = get(f"https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=EW&date={d:%Y/%m/%d}&response=json")
    if j is None: return None
    out = {}
    for t in j.get("tables") or []:
        for r in t.get("data") or []:
            c = four(r[0]) if r else None
            if not c or len(r) < 24: continue
            # 欄位位置：外資(不含自營)2-4、外資自營5-7、外資合計8-10、投信11-13、自營(自行)14-16、自營(避險)17-19、自營合計20-22、三大法人23
            out[c] = {"foreign": lots(r[10]), "trust": lots(r[13]), "dealer": lots(r[22]), "inst": lots(r[23])}
    return out

def twse_margin(d):
    j = get(f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d:%Y%m%d}&selectType=ALL&response=json")
    if j is None: return None
    out = {}
    for t in j.get("tables") or []:
        f = t.get("fields") or []
        if not f or f[0] != "代號": continue
        for r in t.get("data") or []:
            c = four(r[0])
            if c and len(r) >= 13: out[c] = {"margin": num(r[6]), "short": num(r[12])}  # 今日餘額（張）
    return out

def tpex_margin(d):
    j = get(f"https://www.tpex.org.tw/www/zh-tw/margin/balance?date={d:%Y/%m/%d}&response=json")
    if j is None: return None
    out = {}
    for t in j.get("tables") or []:
        f = t.get("fields") or []
        if "資餘額" not in f or "券餘額" not in f: continue
        km, ks = f.index("資餘額"), f.index("券餘額")
        for r in t.get("data") or []:
            c = four(r[0])
            if c: out[c] = {"margin": num(r[km]), "short": num(r[ks])}
    return out

def one_day(d):
    rows, fails = {}, []
    for mk, fns in (("TWSE", (twse_inst, twse_margin)), ("TPEX", (tpex_inst, tpex_margin))):
        for fn in fns:
            res = fn(d); time.sleep(3.5)
            if res is None: fails.append(fn.__name__); continue
            for c, v in res.items():
                rows.setdefault((mk, c), {}).update(v)
    out = [[d.isoformat(), mk, c] + [v.get(k) for k in COLS[3:]] for (mk, c), v in sorted(rows.items())]
    return out, fails

def trading_days(a, b):
    ds = set()
    for fn in sorted(glob.glob("data/history/*.csv.gz")) + ["data/prices.csv"]:
        if not os.path.exists(fn): continue
        y = os.path.basename(fn)[:4]
        if fn.endswith(".gz") and not (a.isoformat()[:4] <= y <= b.isoformat()[:4]): continue
        with (gzip.open(fn, "rt", encoding="utf-8") if fn.endswith(".gz") else open(fn, encoding="utf-8")) as f:
            for r in csv.DictReader(f):
                if r["market"] == "TWSE" and a.isoformat() <= r["date"] <= b.isoformat(): ds.add(r["date"])
    return sorted(dt.date.fromisoformat(x) for x in ds)

def read(fn):
    if not os.path.exists(fn): return []
    with gzip.open(fn, "rt", encoding="utf-8") as f: return [r for r in csv.reader(f)][1:]

def write(fn, rows):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with gzip.open(fn, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(sorted(rows, key=lambda r: (r[0], r[1], r[2])))

def main(start=None, end=None, out=None):
    today = dt.date.today()
    if start:
        a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        fn = out or f"data/chips/{a.year}.csv.gz"; rows = read(fn); have = {r[0] for r in rows}
    else:
        a, b = dt.date(today.year, 1, 1), today
        fn = f"data/chips/{today.year}.csv.gz"; rows = read(fn)
        # 最近3個交易日每次都重抓（融資融券通常晚上才公布，早一點跑可能還沒有）
        recent = trading_days(a, b)[-3:]
        have = {r[0] for r in rows} - {d.isoformat() for d in recent}
        rows = [r for r in rows if r[0] in have]
    failed = []
    for n, d in enumerate(trading_days(a, b)):
        if d.isoformat() in have: continue
        got, fails = one_day(d)
        rows += got
        if fails: failed.append(f"{d} {','.join(fails)}")
        if n % 20 == 0: print(d, len(rows), "rows", flush=True)
    write(fn, rows)
    if failed:
        with open(fn + ".failed.txt", "a") as f: f.write("\n".join(failed) + "\n")
    print("done", fn, len(rows), "rows;", len(failed), "days with failures")

if __name__ == "__main__":
    main(*sys.argv[1:4])
