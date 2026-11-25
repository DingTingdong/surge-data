"""探測：期交所選擇權歷史資料能抓多少年（三大法人買賣權分計、Put/Call 比、每日行情）。只印結果。"""
import io, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0"}
def get(url, data):
    try:
        r = requests.post(url, data=data, headers=H, timeout=90); t = r.content.decode("cp950", errors="replace")
        return r.status_code, t
    except Exception as e: return None, str(e)
tests = [
 ("三大法人買賣權分計", "https://www.taifex.com.tw/cht/3/callsAndPutsDateDown", lambda a, b: {"queryStartDate": a, "queryEndDate": b, "commodityId": "TXO"}),
 ("PutCall比", "https://www.taifex.com.tw/cht/3/pcRatioDown", lambda a, b: {"queryStartDate": a, "queryEndDate": b}),
 ("選擇權每日行情", "https://www.taifex.com.tw/cht/3/optDataDown", lambda a, b: {"down_type": "1", "commodity_id": "TXO", "queryStartDate": a, "queryEndDate": b}),
]
for name, url, mk in tests:
    for a, b in (("2007/07/02", "2007/07/06"), ("2012/03/01", "2012/03/05"), ("2017/06/01", "2017/06/05"), ("2023/01/03", "2023/01/06"), ("2026/11/18", "2026/11/24")):
        st, t = get(url, mk(a, b)); lines = [x for x in t.splitlines() if x.strip()]
        print("##", name, a, "status", st, "行數", len(lines), "| 首行", lines[0][:160] if lines else "", "| 第二行", lines[1][:200] if len(lines) > 1 else "")
        time.sleep(3)
# 一次能抓多長的區間
st, t = get(tests[0][1], tests[0][2]("2020/01/02", "2022/12/30")); print("## 三大法人 三年區間 行數", len([x for x in t.splitlines() if x.strip()]), t[:200].replace("\n", " | "))
st, t = get(tests[2][1], tests[2][2]("2020/01/02", "2020/03/31")); print("## 每日行情 三個月 行數", len([x for x in t.splitlines() if x.strip()]), t[:200].replace("\n", " | "))
