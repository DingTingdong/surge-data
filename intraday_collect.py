"""盤中背景收集（不發任何提醒）：每 3 分鐘掃一次全部上市櫃四碼股票（證交所即時報價），
記下每檔「第一次」漲到 +5%、+7%、漲停的時間、價格、累計量，以及開盤價。收盤後存 data/intraday/YYYY-MM-DD.csv。
11/23 加：每 15 分鐘把全部股票最新價寫到 docs/live.json 推上去，網站盤中顯示（收盤後由每日更新換成正式資料）。
之後用來研究：盤中提早買，會不會比隔天開盤買好。
由 .github/workflows/intraday.yml 每個交易日早上啟動，13:35 左右自己結束；非交易日（報價日期不是今天）直接結束。
"""
import csv, json, os, subprocess, time, datetime as dt, requests

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

def publish(day, latest):
    """盤中把最新報價寫到 docs/live.json 並推上去（網站每幾分鐘讀一次）；推不上去就算了，下次再推。"""
    os.makedirs("docs", exist_ok=True)
    with open("docs/live.json", "w", encoding="utf-8") as f:
        json.dump({"d": day, "t": now().strftime("%H:%M"), "q": latest}, f, ensure_ascii=False, separators=(",", ":"))
    sh = lambda *a: subprocess.run(list(a), capture_output=True, text=True)
    sh("git", "add", "docs/live.json")
    if sh("git", "diff", "--cached", "--quiet").returncode == 0: return
    sh("git", "commit", "-q", "-m", f"live {day} {now().strftime('%H:%M')}")
    for k in range(4):
        if sh("git", "pull", "--rebase", "--autostash", "-q").returncode == 0 and sh("git", "push", "-q").returncode == 0: return
        time.sleep(15)

def main():
    s = requests.Session()
    try: s.get("https://mis.twse.com.tw/stock/index.jsp", headers=H, timeout=20)
    except Exception: pass
    all_codes = codes(); today = now().strftime("%Y%m%d"); day = now().strftime("%Y-%m-%d")
    while now().strftime("%H%M") < "0901": time.sleep(20)
    probe = fetch(s, ["tse_2330.tw"])
    if not probe or probe[0].get("d") != today:
        print("not a trading day", probe[0].get("d") if probe else None); return
    rows, seen, opens, latest = {}, set(), {}, {}
    last_save = time.time(); last_pub = 0
    while now().strftime("%H%M") <= "1333":
        t0 = time.time()
        for i in range(0, len(all_codes), 150):
            for a in fetch(s, all_codes[i:i + 150]):
                c, y, z, u, o, v, h = (a.get("c"), num(a.get("y")), num(a.get("z")), num(a.get("u")), num(a.get("o")), num(a.get("v")), num(a.get("h")))
                if not c or not y: continue
                if o and c not in opens: opens[c] = o
                px = z if z is not None else (latest.get(c, [None])[0])
                if px is None:  # 剛開盤還沒成交或漲停鎖住時 z 可能是「-」：用最佳買價
                    bb = str(a.get("b", "")).split("_")[0]; px = num(bb)
                if px: latest[c] = [px, round((px / y - 1) * 100, 2), v, 1 if (u and px >= u) else 0]
                hi = max([x for x in (z, h) if x is not None], default=None)  # 用當日最高價判斷「碰到」（兩次掃描之間碰到也算）
                if hi is None: continue
                hits = [k for k, th in LEVELS if hi >= y * (1 + th)] + (["limit"] if u and hi >= u else [])
                for k in hits:
                    if (c, k) in seen: continue
                    seen.add((c, k))
                    rows[(c, k)] = [day, c, a.get("n", ""), a.get("ex", ""), y, opens.get(c, ""), k, now().strftime("%H:%M"), z if z is not None else "", v]
            time.sleep(0.5)
        if time.time() - last_save > 1800: save(day, rows); last_save = time.time()
        if time.time() - last_pub > 900: publish(day, latest); last_pub = time.time()
        time.sleep(max(5, 180 - (time.time() - t0)))
    save(day, rows); publish(day, latest); print("saved", len(rows), "events")

if __name__ == "__main__":
    main()
