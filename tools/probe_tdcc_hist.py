"""探測：集保股權分散表（千張大戶）歷史能抓多少年。只印結果。"""
import re, socket, time, requests
import urllib3.util.connection as _uc
_uc.allowed_gai_family = lambda: socket.AF_INET
H = {"User-Agent": "Mozilla/5.0"}
# 1) FinMind（免 token）
for ds in ("TaiwanStockHoldingSharesPer",):
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params={"dataset": ds, "data_id": "2330", "start_date": "2005-01-01", "end_date": "2026-11-25"}, headers=H, timeout=90)
        j = r.json(); d = j.get("data", [])
        print("## FinMind", ds, "status", r.status_code, j.get("msg"), "筆數", len(d), "最早", d[0]["date"] if d else None, "最晚", d[-1]["date"] if d else None, "欄位", list(d[0].keys()) if d else None, "樣本", d[:2])
    except Exception as e: print("## FinMind 失敗", e)
    time.sleep(3)
# 2) 集保查詢頁可選的日期
try:
    s = requests.Session(); r = s.get("https://www.tdcc.com.tw/portal/zh/smWeb/qryStock", headers=H, timeout=60)
    ds = re.findall(r'<option value="(\d{8})"', r.text); print("## 集保查詢頁 status", r.status_code, "可選日期數", len(ds), "最早", ds[-1] if ds else None, "最晚", ds[0] if ds else None)
except Exception as e: print("## 集保查詢頁 失敗", e)
