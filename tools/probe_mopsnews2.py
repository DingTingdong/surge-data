"""探測：重大訊息依日期查詢的 HTML 裡，可轉債公告的明細格式（研究用；只印結果）。"""
import io, re, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
def get(typek, y, m, d):
    r = requests.post("https://mopsov.twse.com.tw/mops/web/ajax_t05st02", data={"encodeURIComponent": 1, "step": 0, "firstin": 1, "TYPEK": typek, "year": y, "month": m, "day": d}, headers=H, timeout=90); r.encoding = "utf-8"; return r.text
for typek, y, m, d in (("sii", "104", "06", "10"), ("otc", "114", "12", "05"), ("sii", "097", "03", "12")):
    txt = get(typek, y, m, d)
    try: t = [x for x in pd.read_html(io.StringIO(txt)) if len(x) > 2][0]; print("##", typek, y, m, d, "rows", len(t))
    except Exception as e: print("##", typek, y, m, d, "no table", len(txt)); continue
    subj = t.iloc[:, 4].astype(str); hit = t[subj.str.contains("轉換公司債") & subj.str.contains("轉換價格")]
    print("## 轉換價格 公告", len(hit), hit.iloc[:, [2, 3, 4]].head(3).values.tolist())
    i = txt.find("轉換價格")
    if i > 0:
        seg = re.sub(r"\s+", " ", txt[max(0, i - 2500): i + 2500]); print("## HTML片段", seg[:5000])
    time.sleep(6)
