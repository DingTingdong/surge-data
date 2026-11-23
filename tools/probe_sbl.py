"""探測：借券賣出餘額（TWSE TWT93U、TPEx 融資融券／借券）歷史資料能抓到哪一年（研究用；只印結果，不寫檔）。"""
import json, time, requests
H = {"User-Agent": "Mozilla/5.0"}
def show(lab, r):
    try:
        j = r.json(); keys = list(j.keys())[:12]
        tabs = j.get("tables") or []
        fields = (j.get("fields") or (tabs[0].get("fields") if tabs else None))
        data = (j.get("data") or (tabs[0].get("data") if tabs else None) or j.get("aaData") or [])
        print("##", lab, r.status_code, "keys", keys, "stat", j.get("stat"), "fields", fields, "rows", len(data), "first", data[:2], flush=True)
    except Exception as e:
        print("##", lab, r.status_code, "not json", r.text[:200].replace("\n", " "), e, flush=True)
for d in ("20051003", "20060301", "20100301", "20150601", "20261120"):
    for url in (f"https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U?date={d}&response=json",
                f"https://www.twse.com.tw/exchangeReport/TWT93U?response=json&date={d}"):
        try: show(f"TWSE {d} {url.split('/')[-1][:30]}", requests.get(url, headers=H, timeout=30))
        except Exception as e: print("##", d, "ERR", e)
        time.sleep(4)
for d in ("096/03/01", "099/03/01", "104/06/01", "115/11/20"):
    for url in (f"https://www.tpex.org.tw/web/stock/margin_trading/margin_sbl/margin_sbl_result.php?l=zh-tw&d={d}&o=json",
                f"https://www.tpex.org.tw/www/zh-tw/margin/sbl?date={d.replace('/', '%2F')}&response=json"):
        try: show(f"TPEx {d} {url.split('/')[-1][:30]}", requests.get(url, headers=H, timeout=30))
        except Exception as e: print("##", d, "ERR", e)
        time.sleep(4)
