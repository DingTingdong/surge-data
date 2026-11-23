"""探測（第 2 次）：內部人持股轉讓事前申報的日期參數格式。"""
import io, re, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
base = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y", "TYPEK": "sii"}
tries = [("t56sb12", {"date1": "1040601", "date2": "1040605"}), ("t56sb12", {"sdate": "104/06/01", "edate": "104/06/05"}),
         ("t56sb12", {"year": "104", "month": "06", "b_date": "1", "e_date": "5"}), ("t56sb12", {"s_date": "1040601", "e_date": "1040605"}),
         ("t56sb12", {"date1": "104/06/01"}), ("t56sb12", {"qryDate": "104/06/01"}),
         ("t56sb12_q1", {"year": "104", "month": "06", "day": "01", "co_id": ""}), ("t56sb12_q1", {"year": "104", "month": "6", "day": "2"})]
for api, prm in tries:
    try:
        r = requests.post(f"https://mopsov.twse.com.tw/mops/web/ajax_{api}", data={**base, **prm}, headers=H, timeout=60); r.encoding = "utf-8"; txt = r.text
        body = re.sub(r"\s+", " ", re.sub(r"<script.*?</script>", " ", txt, flags=re.S)); body = re.sub(r"<[^>]+>", " ", body)
        print("##", api, prm, r.status_code, len(txt), "|", re.sub(r"\s+", " ", body)[:300], flush=True)
        if api == "t56sb12_q1" and prm.get("day") == "01": print("RAW", txt[:1500].replace("\n", " "), flush=True)
    except Exception as e: print("##", api, "ERR", e, flush=True)
    time.sleep(3)
