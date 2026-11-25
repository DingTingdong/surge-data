"""探測：三大法人選擇權買賣權分計的歷史上限（下載檔與每日網頁）。只印結果。"""
import io, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0"}
D = "https://www.taifex.com.tw/cht/3/callsAndPutsDateDown"; P = "https://www.taifex.com.tw/cht/3/callsAndPutsDate"
for a, b in (("2025/11/03", "2025/11/07"), ("2024/06/03", "2024/06/07"), ("2023/12/01", "2023/12/05"), ("2023/11/27", "2023/12/01"), ("2026/01/02", "2026/11/24")):
    t = requests.post(D, data={"queryStartDate": a, "queryEndDate": b, "commodityId": "TXO"}, headers=H, timeout=90).content.decode("cp950", errors="replace")
    print("## 下載", a, b, "CSV" if t.startswith("日期") else "HTML", len(t.splitlines())); time.sleep(3)
for q in ("2007/07/02", "2012/03/01", "2017/06/01", "2023/01/03"):
    for data in ({"queryType": "1", "goDay": "", "doQuery": "1", "dateaddcnt": "", "queryDate": q, "commodityId": "TXO"}, {"queryDate": q, "commodityId": "TXO"}):
        r = requests.post(P, data=data, headers=H, timeout=90); t = r.text
        try:
            tb = [x for x in pd.read_html(io.StringIO(t)) if x.shape[0] >= 6]; print("## 網頁", q, "表格", [x.shape for x in tb][:3], "外資" in t, str(tb[0].iloc[:3, :8].values.tolist())[:300] if tb else "")
        except Exception as e: print("## 網頁", q, "沒有表格", len(t), "外資" in t)
        time.sleep(3)
