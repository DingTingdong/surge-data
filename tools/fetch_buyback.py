"""庫藏股買回（公開資訊觀測站 ajax_t35sc09，依董事會決議日期區間逐年查）2005–今年 → data/buyback/buyback.csv.gz（研究用）。"""
import io, os, re, time, datetime as dt, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
os.makedirs("data/buyback", exist_ok=True); rows, fails = [], []
for y in range(2005, dt.date.today().year + 1):
    for typ in ("sii", "otc"):
        data = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y", "TYPEK": typ, "d1": f"{y - 1911}0101", "d2": f"{y - 1911}1231"}
        txt = ""
        for k in range(3):
            try:
                r = requests.post("https://mopsov.twse.com.tw/mops/web/ajax_t35sc09", data=data, headers=H, timeout=90); r.encoding = "utf-8"; txt = r.text; break
            except Exception: time.sleep(40 * (k + 1))
        try: tabs = [t for t in pd.read_html(io.StringIO(txt)) if any("代號" in str(c) for c in t.columns)]
        except Exception: tabs = []
        if not txt: fails.append(f"{y} {typ}")
        n = 0
        for t in tabs:
            t.columns = ["_".join(dict.fromkeys(str(x).replace(" ", "") for x in (c if isinstance(c, tuple) else (c,)))) for c in t.columns]
            cc = [c for c in t.columns if "代號" in c][0]; t = t[t[cc].astype(str).str.fullmatch(r"\d{4}")]
            t.insert(0, "market", "TWSE" if typ == "sii" else "TPEX"); rows.append(t); n += len(t)
        print(y, typ, n, flush=True); time.sleep(8)
out = pd.concat(rows, ignore_index=True).drop_duplicates() if rows else pd.DataFrame()
out.to_csv("data/buyback/buyback.csv.gz", index=False); print("total", len(out), "fails", fails)
