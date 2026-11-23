"""探測：內部人持股轉讓事前申報，上櫃要用哪個 report 代碼（研究用；只印結果，查 6 次）。"""
import io, re, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
q = requests.post("https://mopsov.twse.com.tw/mops/web/ajax_t56sb12_q1", data={"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y", "TYPEK": "otc", "year": "114", "month": "06", "day": "10"}, headers=H, timeout=60); q.encoding = "utf-8"
print("## q1 otc hidden", re.findall(r"name='(\w+)' value='([^']*)'", q.text)[:12])
mk = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code").market.to_dict()
for rep in ("SO", "OY", "OT", "SS", "TY", "OO"):
    r = requests.post("https://mopsov.twse.com.tw/mops/web/ajax_t56sb12", data={"run": "", "step": "2", "year": "114", "month": "06", "day": "10", "report": rep, "firstin": "true"}, headers=H, timeout=60); r.encoding = "utf-8"
    try: t = [x for x in pd.read_html(io.StringIO(r.text)) if any("申報日期" in str(c) for c in x.columns)]
    except Exception: t = []
    codes = []
    for x in t: cc = [c for c in x.columns if "代號" in str(c)]; codes += x[cc[0]].astype(str).str.strip().tolist() if cc else []
    print("##", rep, len(codes), pd.Series([mk.get(c, "?") for c in codes]).value_counts().to_dict() if codes else {}, flush=True)
    time.sleep(6)
