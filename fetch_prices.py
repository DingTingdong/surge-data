"""每日抓取證交所(上市)與櫃買中心(上櫃)的四碼個股日行情，累積成 data/prices.csv。
第一次執行會從 START 補到今天；之後只補缺的日期。"""
import csv, os, re, time, datetime as dt, requests

START = dt.date(2026, 5, 1)
OUT, HOL = "data/prices.csv", "data/holidays.txt"
COLS = ["date", "market", "code", "name", "open", "high", "low", "close", "volume", "amount"]
H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}

def num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return None

def get(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, headers=H, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(4 * (i + 1))
    return None  # 連線失敗(不是休市)

def idx(fields, key):
    for i, f in enumerate(fields):
        if key in f:
            return i
    return None

def keep(market, d, code, name, o, h, l, c, v, a):
    code = str(code).strip()
    if re.fullmatch(r"\d{4}", code) and num(c) is not None:
        return [d.isoformat(), market, code, str(name).strip(), num(o), num(h), num(l), num(c), num(v), num(a)]

def twse(d):
    j = get(f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={d:%Y%m%d}&type=ALLBUT0999")
    if j is None:
        return None
    tabs = j.get("tables") or [{"fields": j.get("fields9"), "data": j.get("data9")}]
    out = []
    for t in tabs:
        f, data = t.get("fields") or [], t.get("data") or []
        if idx(f, "證券代號") is None or idx(f, "收盤價") is None:
            continue
        k = [idx(f, x) for x in ["證券代號", "證券名稱", "開盤價", "最高價", "最低價", "收盤價", "成交股數", "成交金額"]]
        for r in data:
            row = keep("TWSE", d, r[k[0]], r[k[1]], r[k[2]], r[k[3]], r[k[4]], r[k[5]], r[k[6]], r[k[7]])
            if row:
                out.append(row)
    return out

def tpex(d):
    j = get(f"https://www.tpex.org.tw/www/zh-tw/afterTrading/otc?date={d:%Y/%m/%d}&type=EW&response=json")
    out = []
    if j is not None and j.get("tables"):
        t = j["tables"][0]
        f, data = t.get("fields") or [], t.get("data") or []
        k = [idx(f, x) for x in ["代號", "名稱", "開盤", "最高", "最低", "收盤", "成交股數", "成交金額"]]
        if None in k:
            k = [0, 1, 4, 5, 6, 2, 7, 8]
        for r in data:
            row = keep("TPEX", d, r[k[0]], r[k[1]], r[k[2]], r[k[3]], r[k[4]], r[k[5]], r[k[6]], r[k[7]])
            if row:
                out.append(row)
        return out
    j = get(f"https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&d={d.year-1911}/{d:%m/%d}&o=json")
    if j is None:
        return None
    for r in j.get("aaData") or j.get("tables", [{}])[0].get("data", []):
        row = keep("TPEX", d, r[0], r[1], r[4], r[5], r[6], r[2], r[8], r[9])
        if row:
            out.append(row)
    return out

def main():
    os.makedirs("data", exist_ok=True)
    have = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                have.add((r["date"], r["market"]))
    hol = set(open(HOL).read().split()) if os.path.exists(HOL) else set()
    new = not os.path.exists(OUT)
    today, d = dt.date.today(), START
    with open(OUT, "a", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        if new:
            w.writerow(COLS)
        while d <= today:
            if d.weekday() < 5 and d.isoformat() not in hol:
                got = 0
                for m, fn in (("TWSE", twse), ("TPEX", tpex)):
                    if (d.isoformat(), m) in have:
                        continue
                    rows = fn(d)
                    time.sleep(3.5)
                    if rows:
                        w.writerows(rows); fo.flush(); got += len(rows)
                        print(d, m, len(rows), flush=True)
                    elif rows is not None and d < today - dt.timedelta(days=1):
                        hol.add(d.isoformat()) if m == "TPEX" and got == 0 and not any((d.isoformat(), x) in have for x in ("TWSE", "TPEX")) else None
            d += dt.timedelta(days=1)
    open(HOL, "w").write("\n".join(sorted(hol)))

if __name__ == "__main__":
    main()
