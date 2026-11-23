"""內部人持股轉讓事前申報（公開資訊觀測站 ajax_t56sb12，逐日）→ out/insider-YYYY.csv.gz（研究用；由 insider.yml 平行回補、合併到 data/insider/）。
環境變數 YEAR。每個平日查上市、上櫃各一次（非交易日查無資料就跳過）。"""
import io, os, re, sys, time, datetime as dt, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
Y = int(os.environ.get("YEAR", "2015")); os.makedirs("out", exist_ok=True)
def post(api, data):
    for k in range(3):
        try:
            r = requests.post(f"https://mopsov.twse.com.tw/mops/web/{api}", data=data, headers=H, timeout=40); r.encoding = "utf-8"; return r.text
        except Exception: time.sleep(30 * (k + 1))   # 被擋時多等一下
    return ""
def report_code(typ):
    txt = post("ajax_t56sb12_q1", {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y", "TYPEK": typ, "year": str(Y - 1911), "month": "01", "day": "05"})
    m = re.search(r"name='report' value='([^']*)'", txt); return m.group(1) if m else None
# 11/23：查詢頁回傳的 report 代碼不分市場（都是 SY），上櫃試 SO；回傳的列另外記下 report，之後依代號判斷市場
reps = {"sii": "SY", "otc": "SO"}; print("report codes", reps, "(q1 回傳：", report_code("sii"), ")", flush=True)
rows, fails = [], []
d = dt.date(Y, 1, 1)
while d.year == Y and d <= dt.date.today():
    if d.weekday() < 5:
        for typ, rep in reps.items():
            if not rep: continue
            txt = post("ajax_t56sb12", {"run": "", "step": "2", "year": str(Y - 1911), "month": f"{d.month:02d}", "day": f"{d.day:02d}", "report": rep, "firstin": "true"})
            if not txt: fails.append(f"{d} {typ}"); continue
            try: tabs = [t for t in pd.read_html(io.StringIO(txt)) if len(t) > 0 and any("申報日期" in str(c) for c in t.columns)]
            except Exception: tabs = []
            for t in tabs:
                t.columns = ["_".join(dict.fromkeys(str(x).replace(" ", "") for x in (c if isinstance(c, tuple) else (c,)))) for c in t.columns]
                t = t[t[[c for c in t.columns if "代號" in c][0]].astype(str).str.fullmatch(r"\d{4}")]
                t.insert(0, "report", rep); t.insert(0, "qdate", d.isoformat()); rows.append(t)
            time.sleep(4)
        if d.day in (10, 20, 28) and rows:  # 定期先存（超時也保得住）
            pd.concat(rows, ignore_index=True).to_csv(f"out/insider-{Y}.csv.gz", index=False); open(f"out/insider-{Y}.failed.txt", "w").write("\n".join(fails))
    d += dt.timedelta(days=1)
out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
out.to_csv(f"out/insider-{Y}.csv.gz", index=False); print(Y, "rows", len(out), "fails", len(fails), flush=True)
open(f"out/insider-{Y}.failed.txt", "w").write("\n".join(fails))
