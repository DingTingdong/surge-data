"""抓觀測站「重大訊息依日期查詢」（ajax_t05st02）裡跟可轉債、現金增資有關的公告（含明細全文）到 out/mopsnews-YEAR.csv.gz（研究用；網站不讀）。
每個交易日上市、上櫃各一次查詢；主旨含可轉債、現金增資、法說會、自結損益、訂單／合約、減資、私募等關鍵字的才留（同樣的查詢次數，多存幾類之後研究用）。欄位：date, time, code, name, subject, detail, market。"""
import datetime as dt, html, io, os, re, time, requests, pandas as pd
YEAR = int(os.environ["YEAR"]); os.makedirs("out", exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}; KEYS = ("轉換公司債", "轉換價格", "可轉換", "現金增資", "法人說明會", "自結", "訂單", "合約", "減資", "私募")
def get(typek, d):
    data = {"encodeURIComponent": 1, "step": 0, "firstin": 1, "TYPEK": typek, "year": f"{d.year - 1911:03d}", "month": f"{d.month:02d}", "day": f"{d.day:02d}"}
    for k in range(4):
        try:
            r = requests.post("https://mopsov.twse.com.tw/mops/web/ajax_t05st02", data=data, headers=H, timeout=120); r.encoding = "utf-8"
            if r.status_code == 200 and ("查無" in r.text or "<form" in r.text): return r.text
        except Exception: pass
        time.sleep(30 * (k + 1))
    return None
def parse(txt, mkt):
    rows = []
    for f in re.findall(r"<form action='/mops/web/ajax_t05st02'.*?</form>", txt, flags=re.S):
        vals = re.findall(r"name='h\d+' value='([^']*)'", f)
        if len(vals) < 9: continue
        name, code, d0, tm, subj = [html.unescape(x).strip() for x in vals[:5]]; detail = html.unescape(vals[8]).strip()
        if any(k in subj for k in KEYS): rows.append({"date": d0, "time": tm, "code": code, "name": name, "subject": subj, "detail": re.sub(r"\s+", " ", detail), "market": mkt})
    return rows
rows, failed = [], []; d = dt.date(YEAR, 1, 1); end = min(dt.date(YEAR, 12, 31), dt.date.today())
while d <= end:
    if d.weekday() < 5:
        for typek in ("sii", "otc"):
            txt = get(typek, d)
            if txt is None: failed.append(f"{typek} {d}")
            elif "<form" in txt: rows += parse(txt, typek)
            time.sleep(5)
    d += dt.timedelta(days=1)
df = pd.DataFrame(rows).drop_duplicates(subset=["date", "time", "code", "subject"]) if rows else pd.DataFrame(columns=["date", "time", "code", "name", "subject", "detail", "market"]); df.to_csv(f"out/mopsnews-{YEAR}.csv.gz", index=False, compression="gzip"); open(f"out/mopsnews-{YEAR}.failed.txt", "w").write("\n".join(failed))
print(YEAR, "rows", len(df), "轉換價格", int(df.subject.str.contains("轉換價格").sum()) if len(df) else 0, "現金增資", int(df.subject.str.contains("現金增資").sum()) if len(df) else 0, "failed", len(failed))
