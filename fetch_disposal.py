"""抓證交所、櫃買中心公布的處置股票，存成 data/disposal.csv（四碼代號、處置起迄日）。

用法：python fetch_disposal.py            → 補到今天（從檔案最後一筆往前60天重抓）
      python fetch_disposal.py 2020-01-01 → 從指定日開始，逐月抓
"""
import csv, os, re, sys, time, datetime as dt, requests

OUT = "data/disposal.csv"
COLS = ["market", "code", "name", "announced", "start", "end", "measure"]
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

def roc(s):
    """'115/11/17' → '2026-11-17'"""
    m = re.match(r"\s*(\d{2,3})/(\d{1,2})/(\d{1,2})", str(s))
    return dt.date(int(m[1]) + 1911, int(m[2]), int(m[3])).isoformat() if m else None

def span(s):
    parts = re.split(r"[～~]", str(s))
    return (roc(parts[0]), roc(parts[1])) if len(parts) == 2 else (None, None)

def clean(name):
    return re.sub(r"\(.*?\)|<[^>]+>", "", str(name)).strip()

def twse(a, b):
    j = get(f"https://www.twse.com.tw/rwd/zh/announcement/punish?startDate={a:%Y%m%d}&endDate={b:%Y%m%d}&response=json")
    if j is None: return None
    f = j.get("fields") or []
    if not f: return []
    k = {x: f.index(x) for x in ("公布日期", "證券代號", "證券名稱", "處置起迄時間", "處置措施") if x in f}
    out = []
    for r in j.get("data") or []:
        s, e = span(r[k["處置起迄時間"]])
        out.append(["TWSE", str(r[k["證券代號"]]).strip(), clean(r[k["證券名稱"]]), roc(r[k["公布日期"]]), s, e, r[k["處置措施"]]])
    return out

def tpex(a, b):
    j = get(f"https://www.tpex.org.tw/www/zh-tw/bulletin/disposal?startDate={a:%Y/%m/%d}&endDate={b:%Y/%m/%d}&response=json")
    if j is None: return None
    out = []
    for t in j.get("tables") or []:
        f = t.get("fields") or []
        if "證券代號" not in f: continue
        k = {x: f.index(x) for x in ("公布日期", "證券代號", "證券名稱", "處置起訖時間", "處置措施") if x in f}
        for r in t.get("data") or []:
            s, e = span(r[k["處置起訖時間"]])
            out.append(["TPEX", str(r[k["證券代號"]]).strip(), clean(r[k["證券名稱"]]), roc(r[k["公布日期"]]), s, e, r[k["處置措施"]]])
    return out

def main(start=None):
    rows = []
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            rows = [r for r in csv.reader(f)][1:]
    today = dt.date.today()
    if start:
        a = dt.date.fromisoformat(start)
    elif rows:
        a = dt.date.fromisoformat(max(r[3] for r in rows if r[3])) - dt.timedelta(days=60)
    else:
        a = dt.date(2020, 1, 1)
    new, failed = [], []
    while a <= today:
        b = min(dt.date(a.year + (a.month == 12), a.month % 12 + 1, 1) - dt.timedelta(days=1), today)
        for fn in (twse, tpex):
            r = fn(a, b); time.sleep(3.5)
            if r is None: failed.append(f"{fn.__name__} {a}")
            else: new += r
        a = b + dt.timedelta(days=1)
    keep = {tuple(r) for r in rows + new if re.fullmatch(r"\d{4}", r[1]) and r[4] and r[5]}
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(sorted(keep, key=lambda r: (r[3], r[0], r[1])))
    print(len(keep), "disposal records;", "failed:", failed or "none")

if __name__ == "__main__":
    main(*(sys.argv[1:2]))
