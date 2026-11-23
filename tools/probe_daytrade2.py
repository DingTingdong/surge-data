"""探測：TWTB4U 個股表的欄位（研究用；只印結果）。"""
import time, requests
for d in ("20140106", "20261120"):
    j = requests.get(f"https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U?date={d}&selectType=All&response=json", headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()
    for t in j.get("tables") or []: print("##", d, t.get("title"), t.get("fields"), len(t.get("data") or []), (t.get("data") or [])[:2], flush=True)
    time.sleep(4)
