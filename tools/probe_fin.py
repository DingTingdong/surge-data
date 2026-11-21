"""試抓公開資訊觀測站的季財報彙總表（綜合損益表 t163sb04、資產負債表 t163sb05），看格式與欄位。"""
import io, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
for (y, s) in ((105, 1), (115, 2)):
    for typ in ("sii", "otc"):
        for api in ("t163sb04", "t163sb05"):
            u = f"https://mopsov.twse.com.tw/mops/web/ajax_{api}"
            data = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y", "TYPEK": typ, "year": y, "season": f"{s:02d}"}
            try:
                r = requests.post(u, data=data, headers=H, timeout=60); r.encoding = "utf-8"
                tabs = pd.read_html(io.StringIO(r.text))
                big = [t for t in tabs if len(t) > 5]
                print(y, s, typ, api, "HTTP", r.status_code, "表格數", len(tabs), "大表", [(len(t), list(map(str, t.columns))[:12]) for t in big][:3])
            except Exception as e:
                print(y, s, typ, api, "ERR", type(e).__name__, str(e)[:120], r.text[:200] if 'r' in dir() else "")
            time.sleep(4)
