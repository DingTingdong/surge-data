"""探測（第 3 次）：印出 t56sb12_q1 自動送出表單的欄位，再照欄位送 ajax_t56sb12。"""
import io, re, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
base = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y", "TYPEK": "sii"}
r = requests.post("https://mopsov.twse.com.tw/mops/web/ajax_t56sb12_q1", data={**base, "year": "104", "month": "06", "day": "01"}, headers=H, timeout=60); r.encoding = "utf-8"
inputs = re.findall(r"<input[^>]*>", r.text, flags=re.I); print("INPUTS", inputs, flush=True)
fields = {}
for tag in inputs:
    n = re.search(r"name=['\"]?([^'\" >]+)", tag, flags=re.I); v = re.search(r"value=['\"]?([^'\">]*)", tag, flags=re.I)
    if n: fields[n.group(1)] = v.group(1) if v else ""
print("FIELDS", fields, flush=True)
for extra in ({}, {"step": "2"}):
    r2 = requests.post("https://mopsov.twse.com.tw/mops/web/ajax_t56sb12", data={**fields, **extra}, headers=H, timeout=60); r2.encoding = "utf-8"
    try: tabs = [t for t in pd.read_html(io.StringIO(r2.text)) if len(t) > 2]
    except Exception: tabs = []
    body = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", " ", r2.text, flags=re.S)))
    print("POST", extra, r2.status_code, len(r2.text), "tables", len(tabs), [t.shape for t in tabs][:3], (list(map(str, tabs[0].columns))[:8] if tabs else ""), "|", body[:400], flush=True)
    if tabs: print(tabs[0].head(5).to_string()[:1500], flush=True)
    time.sleep(3)
