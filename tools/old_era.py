"""補抓 2015-06 ～ 2019-12 的舊資料，當作從來沒用過的全新檢驗期（研究用，網站不讀）。

輸出到 out/，由 .github/workflows/old-era.yml 合併到 data/old/：
  python tools/old_era.py prices YEAR   → out/hist-YEAR.csv.gz（欄位同 data/history）
  python tools/old_era.py chips YEAR    → out/chips-YEAR.csv.gz（欄位同 data/chips）
  python tools/old_era.py misc          → out/adjust.csv、out/disposal.csv、out/revenue-YYYY.csv.gz

從 2015-06-01 開始：漲跌幅限制在這天由 7% 放寬到 10%，之前的「漲停」定義不同。
舊格式：證交所 T86 在 2017/12 中以前只有「外資買賣超股數」一欄；櫃買法人在 2018/1 中以前只有舊網址、17 欄格式。
"""
import csv, gzip, os, sys, time, datetime as dt
sys.path.insert(0, os.getcwd())
import fetch_history as fh, fetch_chips as fc, fetch_adjust as fa, fetch_disposal as fd, fetch_revenue as fr

START = dt.date(2015, 6, 1); END = dt.date(2019, 12, 31)
os.makedirs("out", exist_ok=True)

def span(year):
    return max(START, dt.date(year, 1, 1)), min(END, dt.date(year, 12, 31))

def twse_inst(d):
    j = fc.get(f"https://www.twse.com.tw/rwd/zh/fund/T86?date={d:%Y%m%d}&selectType=ALLBUT0999&response=json")
    if j is None: return None
    f = [x.replace(" ", "") for x in (j.get("fields") or [])]
    if not f: return {}
    k = lambda n: f.index(n) if n in f else None
    kc, kt, kd, ka = k("證券代號"), k("投信買賣超股數"), k("自營商買賣超股數"), k("三大法人買賣超股數")
    kf_new1, kf_new2, kf_old = k("外陸資買賣超股數(不含外資自營商)"), k("外資自營商買賣超股數"), k("外資買賣超股數")
    if None in (kc, kt, kd, ka) or (kf_old is None and kf_new1 is None): return None
    out = {}
    for r in j.get("data") or []:
        c = fc.four(r[kc])
        if not c: continue
        fo = fc.num(r[kf_old]) if kf_old is not None else (fc.num(r[kf_new1]) or 0) + ((fc.num(r[kf_new2]) or 0) if kf_new2 is not None else 0)
        out[c] = {"foreign": None if fo is None else round(fo / 1000, 3), "trust": fc.lots(r[kt]), "dealer": fc.lots(r[kd]), "inst": fc.lots(r[ka])}
    return out

def tpex_inst(d):
    new = fc.tpex_inst(d)
    if new: return new
    j = fc.get(f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&se=EW&t=D&d={d.year - 1911}/{d:%m/%d}&o=json")
    if j is None: return None if new is None else {}
    out = {}
    for t in j.get("tables") or []:
        f = t.get("fields") or []
        rd = str(t.get("date") or "")
        if rd and rd.replace("/", "").lstrip("0") != f"{d.year - 1911}{d:%m%d}": continue  # 不是要的那天
        if len(f) == 16 and "外資及陸資淨買股數" in f:  # 舊 17 欄格式
            for r in t.get("data") or []:
                c = fc.four(r[0]) if r else None
                if c: out[c] = {"foreign": fc.lots(r[4]), "trust": fc.lots(r[7]), "dealer": fc.lots(r[8]), "inst": fc.lots(r[15])}
        elif len(f) == 24:
            for r in t.get("data") or []:
                c = fc.four(r[0]) if r else None
                if c and len(r) >= 24: out[c] = {"foreign": fc.lots(r[10]), "trust": fc.lots(r[13]), "dealer": fc.lots(r[22]), "inst": fc.lots(r[23])}
    return out

def chips(year):
    a, b = span(year)
    rows, failed = [], []
    d = a
    while d <= b:
        if d.weekday() < 5:
            got = {}
            for mk, fns in (("TWSE", (twse_inst, fc.twse_margin)), ("TPEX", (tpex_inst, fc.tpex_margin))):
                for fn in fns:
                    res = fn(d); time.sleep(3.5)
                    if res is None: failed.append(f"{d} {fn.__name__}"); continue
                    for c, v in res.items(): got.setdefault((mk, c), {}).update(v)
            rows += [[d.isoformat(), mk, c] + [v.get(k) for k in fc.COLS[3:]] for (mk, c), v in sorted(got.items())]
            if d.day <= 7 and d.weekday() == 0: print(d, len(rows), "rows", len(failed), "failed", flush=True)
        d += dt.timedelta(days=1)
    with gzip.open(f"out/chips-{year}.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(fc.COLS); w.writerows(rows)
    open(f"out/chips-{year}.failed.txt", "w").write("\n".join(failed))
    print("chips", year, len(rows), "rows,", len(failed), "failed")

def misc():
    adj, dis, fails = [], [], []
    a = START
    while a <= END:
        b = dt.date(a.year + (a.month == 12), a.month % 12 + 1, 1) - dt.timedelta(days=1)
        for fn in (fa.twse_ex, fa.twse_cut, fa.tpex_ex, fa.tpex_cut):
            r = fn(a, b); time.sleep(3.5)
            if r is None: fails.append(f"adjust {fn.__name__} {a:%Y-%m}")
            else: adj += [[str(x) for x in row] for row in r]
        for fn in (fd.twse, fd.tpex):
            r = fn(a, b); time.sleep(3.5)
            if r is None: fails.append(f"disposal {fn.__name__} {a:%Y-%m}")
            else: dis += [[str(x) for x in row] for row in r]
        a = b + dt.timedelta(days=1)
    for fn, cols, rows in (("out/adjust.csv", fa.COLS, adj), ("out/disposal.csv", fd.COLS, dis)):
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(cols); w.writerows(sorted({tuple(r) for r in rows}))
    print("adjust", len(adj), "disposal", len(dis), flush=True)
    for y in range(2014, 2020):  # 2014 起：2015 年的「近3月營收」要用到
        rows = []
        for m in range(1, 13):
            rs, ok = fr.month_rows(y, m)
            print(f"{y}-{m:02d}", len(rs), "rows" if ok else "FAILED", flush=True)
            if not ok: fails.append(f"revenue {y}-{m:02d}")
            rows += [[str(v) if v is not None else "" for v in r] for r in rs]
        fr.write(f"out/revenue-{y}.csv.gz", rows)
    open("out/misc.failed.txt", "w").write("\n".join(fails))
    print("misc failed:", fails or "none")

if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "prices":
        y = int(sys.argv[2]); a, b = span(y); fh.main(a.isoformat(), b.isoformat(), f"out/hist-{y}.csv.gz")
    elif kind == "chips": chips(int(sys.argv[2]))
    elif kind == "misc": misc()
