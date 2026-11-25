"""探測：fetch_cb_daily 每天抓到幾則重大訊息、含可轉債的幾則（只印數字）。"""
import datetime as dt, re, socket, time, requests
import urllib3.util.connection as _uc
_uc.allowed_gai_family = lambda: socket.AF_INET
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
for d in (dt.date(2026, 11, 24), dt.date(2026, 11, 20), dt.date(2026, 11, 3)):
    r = requests.post("https://mopsov.twse.com.tw/mops/web/ajax_t05st02", data={"encodeURIComponent": 1, "step": 0, "firstin": 1, "TYPEK": "sii", "year": f"{d.year-1911:03d}", "month": f"{d.month:02d}", "day": f"{d.day:02d}"}, headers=H, timeout=90); r.encoding = "utf-8"
    f = re.findall(r"<form action='/mops/web/ajax_t05st02'.*?</form>", r.text, flags=re.S)
    print("##", d, r.status_code, "公告數", len(f), "含轉換公司債", sum("轉換公司債" in x for x in f), "含代收", sum("代收" in x for x in f)); time.sleep(4)
