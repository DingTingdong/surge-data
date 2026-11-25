"""每天抓觀測站重大訊息裡「可轉換公司債…代收」公告（第 57 輪往後記錄用），附加到 data/cb_live.csv（只增不改）。
查最近 4 個工作日；TYPEK=sii 會一起回上市、上櫃公司。規則同 research/cb_holdout.py：排除私募／海外／外國／子公司。"""
import datetime as dt, html, os, re, socket, time, requests, pandas as pd
import urllib3.util.connection as _uc
_uc.allowed_gai_family = lambda: socket.AF_INET
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}; OUT = "data/cb_live.csv"
def get(d):
    data = {"encodeURIComponent": 1, "step": 0, "firstin": 1, "TYPEK": "sii", "year": f"{d.year - 1911:03d}", "month": f"{d.month:02d}", "day": f"{d.day:02d}"}
    for k in range(3):
        try:
            r = requests.post("https://mopsov.twse.com.tw/mops/web/ajax_t05st02", data=data, headers=H, timeout=90); r.encoding = "utf-8"
            if r.status_code == 200 and ("查無" in r.text or "<form" in r.text): return r.text
        except Exception: pass
        time.sleep(15 * (k + 1))
    return None
rows, n = [], 0
d = (dt.datetime.utcnow() + dt.timedelta(hours=8)).date()
while n < 4:
    if d.weekday() < 5:
        n += 1; txt = get(d)
        if txt is None: print("failed", d)
        else:
            for f in re.findall(r"<form action='/mops/web/ajax_t05st02'.*?</form>", txt, flags=re.S):
                v = re.findall(r"name='h\d+' value='([^']*)'", f)
                if len(v) < 9: continue
                name, code, d0, tm, subj = [html.unescape(x).strip() for x in v[:5]]; s = re.sub(r"\s+", "", subj)
                if "轉換公司債" in s and "代收" in s and not re.search("私募|海外|外國|子公司", s):
                    rows.append({"date": d0, "time": tm, "code": code, "name": name, "subject": subj})
        time.sleep(4)
    d -= dt.timedelta(days=1)
old = pd.read_csv(OUT, dtype=str) if os.path.exists(OUT) else pd.DataFrame(columns=["date", "time", "code", "name", "subject"])
new = pd.concat([old, pd.DataFrame(rows, columns=old.columns).astype(str)]).drop_duplicates(["date", "time", "code", "subject"])
new.to_csv(OUT, index=False); print("cb_live", len(old), "→", len(new))
