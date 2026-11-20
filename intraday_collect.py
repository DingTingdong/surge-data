"""盤中背景收集（不發任何提醒）：每 3 分鐘掃一次全部上市櫃四碼股票（證交所即時報價），
記下每檔「第一次」漲到 +5%、+7%、漲停的時間、價格、累計量，以及開盤價。收盤後存 data/intraday/YYYY-MM-DD.csv。
之後用來研究：盤中提早買，會不會比隔天開盤買好。
由 .github/workflows/intraday.yml 每個交易日早上啟動，13:35 左右自己結束；非交易日（報價日期不是今天）直接結束。
"""
import csv, os, time, datetime as dt, requests

TZ = dt.timezone(dt.timedelta(hours=8))
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://mis.twse.com.tw/stock/index.jsp"}
LEVELS = [("5", 0.05), ("7", 0.07)]
COLS = ["date", "code", "name", "market", "prev", "open", "level", "time", "price", "vol"]  # time＝掃描到的時間（誤差 3 分鐘內），price＝那時最新成交價

def now(): return dt.datetime.now(TZ)
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def codes():
    out = []
    with open("data/industry.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f): out.append(("tse" if r["market"] == "TWSE" else "otc") + f"_{r['code']}.tw")
    return out

def fetch(s, ex):
    for k in range(3):
        try:
            u = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={'|'.join(ex)}&json=1&delay=0&_={int(time.time() * 1000)}"
            return s.get(u, headers=H, timeout=20).json().get("msgArray", [])
        except Exception: time.sleep(3 * (k + 1))
    return []

def save(day, rows):
    os.makedirs("data/intraday", exist_ok=True)
    with open(f"data/intraday/{day}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(sorted(rows.values(), key=lambda r: (r[7], r[1])))

def main():
    s = requests.Session()
    try: s.get("https://mis.twse.com.tw/stock/index.jsp", headers=H, timeout=20)
    except Exception: pass
    all_codes = codes(); today = now().strftime("%Y%m%d"); day = now().strftime("%Y-%m-%d")
    while now().strftime("%H%M") < "0901": time.sleep(20)
    probe = fetch(s, ["tse_2330.tw"])
    if not probe or probe[0].get("d") != today:
        print("not a trading day", probe[0].get("d") if probe else None); return
    rows, seen, opens = {}, set(), {}
    last_save = time.time()
    while now().strftime("%H%M") <= "1333":
        t0 = time.time()
        for i in range(0, len(all_codes), 150):
            for a in fetch(s, all_codes[i:i + 150]):
                c, y, z, u, o, v, h = (a.get("c"), num(a.get("y")), num(a.get("z")), num(a.get("u")), num(a.get("o")), num(a.get("v")), num(a.get("h")))
                if not c or not y: continue
                if o and c not in opens: opens[c] = o
                hi = max([x for x in (z, h) if x is not None], default=None)  # 用當日最高價判斷「碰到」（兩次掃描之間碰到也算）
                if hi is None: continue
                hits = [k for k, th in LEVELS if hi >= y * (1 + th)] + (["limit"] if u and hi >= u else [])
                for k in hits:
                    if (c, k) in seen: continue
                    seen.add((c, k))
                    rows[(c, k)] = [day, c, a.get("n", ""), a.get("ex", ""), y, opens.get(c, ""), k, now().strftime("%H:%M"), z if z is not None else "", v]
            time.sleep(0.5)
        if time.time() - last_save > 1800: save(day, rows); last_save = time.time()
        time.sleep(max(5, 180 - (time.time() - t0)))
    save(day, rows); print("saved", len(rows), "events")

if __name__ == "__main__":
    main()
