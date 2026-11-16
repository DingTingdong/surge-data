"""抓官方的除權息與減資資料，整理成 data/adjust.csv：每列 = 某檔股票在某天的價格調整比例。

比例 = 參考價 ÷ 前一個交易日收盤價（除權息：除權息參考價 ÷ 除權息前收盤價；減資：恢復買賣參考價 ÷ 停止買賣前收盤價）。
除息讓比例 < 1，減資彌補虧損讓比例 > 1。radar_core 用它把之前的價格乘上比例，消除假漲跌。

用法：python fetch_adjust.py            → 從檔案最後一筆往前60天補到今天
      python fetch_adjust.py 2020-01-01 → 從指定日逐月抓
"""
import csv, os, re, sys, time, datetime as dt, requests

OUT = "data/adjust.csv"
COLS = ["date", "market", "code", "kind", "before", "ref", "ratio"]
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

def roc(s):
    """'109年06月01日'、'109/06/01'、'1090113' → '2020-06-01'"""
    s = str(s).strip()
    m = re.match(r"(\d{2,3})\D+(\d{1,2})\D+(\d{1,2})", s) or re.fullmatch(r"(\d{3})(\d{2})(\d{2})", s)
    return dt.date(int(m[1]) + 1911, int(m[2]), int(m[3])).isoformat() if m else None

def rows_from(fields, data, market, kind, kd, kc, kb, kr):
    k = {name: fields.index(name) for name in (kd, kc, kb, kr) if name in fields}
    if len(k) < 4: return []
    out = []
    for r in data or []:
        d, c, b, ref = roc(r[k[kd]]), str(r[k[kc]]).strip(), num(r[k[kb]]), num(r[k[kr]])
        if d and re.fullmatch(r"\d{4}", c) and b and ref and b > 0 and ref > 0:
            out.append([d, market, c, kind, b, ref, round(ref / b, 6)])
    return out

def twse_ex(a, b):
    j = get(f"https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate={a:%Y%m%d}&endDate={b:%Y%m%d}&response=json")
    return None if j is None else rows_from(j.get("fields") or [], j.get("data"), "TWSE", "除權息", "資料日期", "股票代號", "除權息前收盤價", "除權息參考價")

def twse_cut(a, b):
    j = get(f"https://www.twse.com.tw/rwd/zh/reducation/TWTAUU?startDate={a:%Y%m%d}&endDate={b:%Y%m%d}&response=json")
    return None if j is None else rows_from(j.get("fields") or [], j.get("data"), "TWSE", "減資", "恢復買賣日期", "股票代號", "停止買賣前收盤價格", "恢復買賣參考價")

def tpex_tables(url, kind, kd, kc, kb, kr):
    j = get(url)
    if j is None: return None
    out = []
    for t in j.get("tables") or []:
        out += rows_from(t.get("fields") or [], t.get("data"), "TPEX", kind, kd, kc, kb, kr)
    return out

def tpex_ex(a, b):
    return tpex_tables(f"https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ?startDate={a:%Y/%m/%d}&endDate={b:%Y/%m/%d}&response=json",
                       "除權息", "除權息日期", "代號", "除權息前收盤價", "除權息參考價")

def tpex_cut(a, b):
    return tpex_tables(f"https://www.tpex.org.tw/www/zh-tw/bulletin/revivt?startDate={a:%Y/%m/%d}&endDate={b:%Y/%m/%d}&response=json",
                       "減資", "恢復買賣日期", "股票代號", "最後交易日之收盤價格", "減資恢復買賣開始日參考價格")

def main(start=None):
    rows = []
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            rows = [r for r in csv.reader(f)][1:]
    today = dt.date.today()
    a = dt.date.fromisoformat(start) if start else (dt.date.fromisoformat(max(r[0] for r in rows)) - dt.timedelta(days=60) if rows else dt.date(2020, 1, 1))
    new, failed = [], []
    while a <= today:
        b = min(dt.date(a.year + (a.month == 12), a.month % 12 + 1, 1) - dt.timedelta(days=1), today)
        for fn in (twse_ex, twse_cut, tpex_ex, tpex_cut):
            r = fn(a, b); time.sleep(3.5)
            if r is None: failed.append(f"{fn.__name__} {a:%Y-%m}")
            else: new += [[str(x) for x in row] for row in r]
        a = b + dt.timedelta(days=1)
    keep = {tuple(r[:4]): r for r in rows + new}  # 同日同股同類以新抓的為準
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(sorted(keep.values()))
    print(len(keep), "adjustment records;", "failed:", failed or "none")

if __name__ == "__main__":
    main(*(sys.argv[1:2]))
