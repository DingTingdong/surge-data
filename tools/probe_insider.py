"""探測：公開資訊觀測站「內部人持股轉讓事前申報」歷史資料能不能依日期查（研究用；只印結果，不寫檔）。"""
import io, re, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
base = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y"}
tries = []
for api in ("t56sb12_q1", "t56sb12_q2", "t56sb12_q3", "t56sb12", "t56sb21_q1", "t56sb21_q2"):
    for typ in ("sii",):
        for prm in ({"year": 104, "month": "06", "day": "01"}, {"year": 104, "month": "06"}, {"year": 104, "month": "6", "day": "1", "TYPEK2": ""}, {"date1": "104/06/01", "date2": "104/06/05"}):
            tries.append((api, {**base, "TYPEK": typ, **prm}))
for api, data in tries:
    try:
        r = requests.post(f"https://mopsov.twse.com.tw/mops/web/ajax_{api}", data=data, headers=H, timeout=60); r.encoding = "utf-8"; txt = r.text
        try: tabs = [t for t in pd.read_html(io.StringIO(txt)) if len(t) > 2]
        except Exception: tabs = []
        snip = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", txt))[:160]
        print(api, {k: v for k, v in data.items() if k not in base}, r.status_code, len(txt), "tables", len(tabs), (tabs[0].shape if tabs else ""), (list(map(str, tabs[0].columns))[:6] if tabs else ""), "|", snip, flush=True)
    except Exception as e: print(api, "ERR", e, flush=True)
    time.sleep(3)
