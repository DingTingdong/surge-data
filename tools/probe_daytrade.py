"""探測：現股當沖（TWSE TWTB4U、TPEx）歷史資料能抓到哪一年（研究用；只印結果）。"""
import time, requests
H = {"User-Agent": "Mozilla/5.0"}
def show(lab, r):
    try:
        j = r.json(); t = j.get("tables") or []
        info = [(x.get("title"), x.get("fields"), len(x.get("data") or []), (x.get("data") or [])[:1]) for x in t] if t else (j.get("fields"), len(j.get("data") or []))
        print("##", lab, r.status_code, j.get("stat"), info, flush=True)
    except Exception as e: print("##", lab, r.status_code, r.text[:150].replace("\n", " "), e, flush=True)
for d in ("20140106", "20140601", "20170601", "20261120"):
    show(f"TWSE {d}", requests.get(f"https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U?date={d}&selectType=All&response=json", headers=H, timeout=30)); time.sleep(4)
for d in ("103/06/02", "106/06/01", "115/11/20"):
    show(f"TPEx {d}", requests.get(f"https://www.tpex.org.tw/web/stock/trading/intraday_trading/intraday_trading_list_result.php?l=zh-tw&d={d}&o=json", headers=H, timeout=30)); time.sleep(4)
