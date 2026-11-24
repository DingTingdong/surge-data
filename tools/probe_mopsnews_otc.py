"""探測：上櫃（TYPEK=otc）重大訊息依日期查詢為什麼幾乎沒抓到（研究用；只印結果，不存檔）。"""
import re, time, requests
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
U = "https://mopsov.twse.com.tw/mops/web/ajax_t05st02"
for typek in ("otc", "OTC", "rotc", "sii"):
    r = requests.post(U, data={"encodeURIComponent": 1, "step": 0, "firstin": 1, "TYPEK": typek, "year": "114", "month": "12", "day": "05"}, headers=H, timeout=120); r.encoding = "utf-8"; t = r.text
    forms = re.findall(r"<form[^>]*>", t)
    print("##", typek, "status", r.status_code, "len", len(t), "查無" in t, "forms", len(forms), forms[:2])
    acts = set(re.findall(r"action=['\"]([^'\"]+)['\"]", t)); print("   actions", list(acts)[:5])
    codes = re.findall(r"name='h1' value='([^']*)'", t)[:10]; print("   h1(code?) 前10", codes)
    i = t.find("<form"); print("   片段", re.sub(r"\s+", " ", t[i:i + 800]))
    time.sleep(8)
