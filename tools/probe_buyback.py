"""探測：公開資訊觀測站庫藏股買回的歷史資料（研究用；只印結果，不寫檔）。"""
import io, re, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
base = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y", "TYPEK": "sii"}
tries = []
for api in ("t35sc09", "t35sc09_q1", "t35sc09_q2", "t35sc10", "t35sc11", "t35sc09_q3"):
    for prm in ({"year": "109", "month": "03"}, {"year": "109", "month": "03", "day": "20"}, {"year": "109"}, {"d1": "1090301", "d2": "1090331"}):
        tries.append((api, prm))
for api, prm in tries:
    try:
        r = requests.post(f"https://mopsov.twse.com.tw/mops/web/ajax_{api}", data={**base, **prm}, headers=H, timeout=60); r.encoding = "utf-8"; txt = r.text
        try: tabs = [t for t in pd.read_html(io.StringIO(txt)) if len(t) > 2]
        except Exception: tabs = []
        inputs = {m.group(1): m.group(2) for m in re.finditer(r"name='([^']+)' value='([^']*)'", txt)}
        body = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", " ", txt, flags=re.S)))[:220]
        print("##", api, prm, r.status_code, len(txt), "tables", len(tabs), [t.shape for t in tabs][:2], (list(map(str, tabs[0].columns))[:6] if tabs else ""), "inputs", inputs, "|", body, flush=True)
    except Exception as e: print("##", api, "ERR", e, flush=True)
    time.sleep(2.5)
