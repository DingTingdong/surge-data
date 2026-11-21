"""每日籌碼補充資料（研究用）：外資持股（含發行股數）、融券與借券賣出餘額、當沖成交（上市）。
  python tools/fetch_extra.py probe 2016-01-04   → 印出解析結果
  python tools/fetch_extra.py year 2016          → out/extra-2016.csv.gz（2015 從 6/1 起；今年到今天）
欄位：date, market, code, shares（發行股數）, fshares（外資持股股數）, fratio（外資持股%）, short_bal（融券餘額，張）,
      sbl_sell（借券當日賣出，股）, sbl_ret（借券當日還券，股）, sbl_bal（借券賣出餘額，股）, dt_shares（當沖成交股數）, dt_buy, dt_sell（當沖買進／賣出金額）
"""
import csv, gzip, os, sys, time, datetime as dt
sys.path.insert(0, os.getcwd())
from fetch_chips import get, four, num
FAILED = []
def gj(u, tag):
    j = get(u)
    if j is None: FAILED.append(tag)
    return j
COLS = ["date", "market", "code", "shares", "fshares", "fratio", "short_bal", "sbl_sell", "sbl_ret", "sbl_bal", "dt_shares", "dt_buy", "dt_sell"]

def idx(fields, *names):
    f = [str(x).replace(" ", "") for x in fields]
    for n in names:
        for i, x in enumerate(f):
            if n in x: return i
    return None

def twse(d):
    out = {}
    j = gj(f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date={d:%Y%m%d}&selectType=ALLBUT0999&response=json", f"{d} twse_qfii"); time.sleep(3.5)
    if j and j.get("fields"):
        f = j["fields"]; ic, ish, ifs, ifr = idx(f, "證券代號"), idx(f, "發行股數"), idx(f, "全體外資及陸資持有股數"), idx(f, "全體外資及陸資持股比率")
        for r in j.get("data") or []:
            c = four(r[ic])
            if c: out.setdefault(c, {}).update(shares=num(r[ish]), fshares=num(r[ifs]), fratio=num(r[ifr]))
    j = gj(f"https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U?date={d:%Y%m%d}&response=json", f"{d} twse_sbl"); time.sleep(3.5)
    if j and j.get("data"):
        for r in j["data"]:
            c = four(r[0])
            if c and len(r) >= 13: out.setdefault(c, {}).update(short_bal=num(r[6]), sbl_sell=num(r[9]), sbl_ret=num(r[10]), sbl_bal=num(r[12]))
    j = gj(f"https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U?date={d:%Y%m%d}&selectType=All&response=json", f"{d} twse_dt"); time.sleep(3.5)
    for t in (j or {}).get("tables") or []:
        f = t.get("fields") or []; ic = idx(f, "證券代號")
        if ic is None: continue
        ish, ib, isl = idx(f, "成交股數"), idx(f, "買進成交金額"), idx(f, "賣出成交金額")
        for r in t.get("data") or []:
            c = four(r[ic])
            if c: out.setdefault(c, {}).update(dt_shares=num(r[ish]) if ish is not None else None, dt_buy=num(r[ib]) if ib is not None else None, dt_sell=num(r[isl]) if isl is not None else None)
    return out

def tpex(d):
    out = {}; r_ = f"{d.year - 1911}/{d:%m/%d}"
    j = gj(f"https://www.tpex.org.tw/www/zh-tw/insti/qfii?date={d:%Y/%m/%d}&response=json", f"{d} tpex_qfii"); time.sleep(3.5)
    for t in (j or {}).get("tables") or []:
        f = t.get("fields") or []; ic, ish, ifs, ifr = idx(f, "代號"), idx(f, "發行股數"), idx(f, "持有股數"), idx(f, "持股比率")
        if ic is None: continue
        for r in t.get("data") or []:
            c = four(r[ic])
            if c: out.setdefault(c, {}).update(shares=num(r[ish]), fshares=num(r[ifs]), fratio=num(r[ifr]))
    j = gj(f"https://www.tpex.org.tw/web/stock/margin_trading/margin_sbl/margin_sbl_result.php?l=zh-tw&d={r_}&o=json", f"{d} tpex_sbl"); time.sleep(3.5)
    for t in (j or {}).get("tables") or []:
        for r in t.get("data") or []:
            c = four(r[0]) if r else None
            if c and len(r) >= 13: out.setdefault(c, {}).update(short_bal=num(r[6]), sbl_sell=num(r[9]), sbl_ret=num(r[10]), sbl_bal=num(r[12]))
    return out

def day_rows(d):
    rows = []
    for mk, fn in (("TWSE", twse), ("TPEX", tpex)):
        for c, v in sorted(fn(d).items()): rows.append([d.isoformat(), mk, c] + [v.get(k) for k in COLS[3:]])
    return rows

if __name__ == "__main__":
    if sys.argv[1] == "probe":
        d = dt.date.fromisoformat(sys.argv[2]); rows = day_rows(d)
        print(len(rows), "rows"); [print(r) for r in rows[:3]]; [print(r) for r in [x for x in rows if x[1] == "TPEX"][:3]]
        print("非空比例", {k: round(sum(r[i] is not None for r in rows) / max(1, len(rows)), 2) for i, k in enumerate(COLS) if i >= 3})
    else:
        y = int(sys.argv[2]); a = max(dt.date(y, 1, 1), dt.date(2015, 6, 1)); b = min(dt.date(y, 12, 31), dt.date.today())
        os.makedirs("out", exist_ok=True); rows = []; d = a
        while d <= b:
            if d.weekday() < 5:
                rr = day_rows(d); rows += rr
                if d.day <= 7 and d.weekday() == 0: print(d, len(rows), flush=True)
            d += dt.timedelta(days=1)
        with gzip.open(f"out/extra-{y}.csv.gz", "wt", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(COLS); w.writerows(rows)
        open(f"out/extra-{y}.failed.txt", "w").write("\n".join(FAILED)); print("done", y, len(rows), "failed", len(FAILED))
