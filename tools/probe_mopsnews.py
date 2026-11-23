"""探測：觀測站重大訊息依日期查詢（ajax_t05st02）的格式，找可轉債轉換價格公告（研究用；只印結果）。"""
import io, re, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
for host in ("https://mopsov.twse.com.tw", "https://mops.twse.com.tw"):
    for prm in ({"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "TYPEK": "all", "year": "104", "month": "06", "day": "10"},
                {"encodeURIComponent": 1, "step": 0, "firstin": 1, "TYPEK": "sii", "year": "104", "month": "06", "day": "10"}):
        try:
            r = requests.post(f"{host}/mops/web/ajax_t05st02", data=prm, headers=H, timeout=60); r.encoding = "utf-8"; txt = r.text
            try: tabs = [t for t in pd.read_html(io.StringIO(txt)) if len(t) > 2]
            except Exception: tabs = []
            body = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", " ", txt, flags=re.S)))
            print("##", host, prm.get("TYPEK"), prm.get("step"), r.status_code, len(txt), "tables", [t.shape for t in tabs][:3], (list(map(str, tabs[0].columns))[:8] if tabs else ""), flush=True)
            if tabs: print("## rows", tabs[0].head(4).to_string()[:800]); hits = [x for x in tabs[0].astype(str).values.tolist() if "轉換" in " ".join(x)]; print("## 轉換", len(hits), hits[:3])
            else: print("## body", body[:300])
            m = re.findall(r"onclick=\"[^\"]*\"", txt)[:2]; print("## onclick", m)
        except Exception as e: print("##", host, "ERR", e, flush=True)
        time.sleep(5)
