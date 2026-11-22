"""季財報彙總（公開資訊觀測站 t163sb04 綜合損益表、t163sb05 資產負債表）2013Q1～最新 → data/fin/YYYYQs.csv.gz（研究用）。
長格式：market, stmt（is／bs）, code, name, field, value。各產業格式不同，全部欄位都保留。
損益表數字是否為「年初累計」要在分析時用 Q1/Q2 比對確認。已存在的季度不重抓。
"""
import io, os, sys, time, datetime as dt, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
os.makedirs("data/fin", exist_ok=True)
def num(x):
    try: return float(str(x).replace(",", ""))
    except Exception: return None
def one(api, typ, y, s):
    data = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y", "TYPEK": typ, "year": y - 1911, "season": f"{s:02d}"}
    for k in range(4):
        try:
            r = requests.post(f"https://mopsov.twse.com.tw/mops/web/ajax_{api}", data=data, headers=H, timeout=90); r.encoding = "utf-8"
            return [t for t in pd.read_html(io.StringIO(r.text)) if len(t) > 3 and any("代號" in str(c) for c in t.columns)]
        except ValueError: return []
        except Exception: time.sleep(10 * (k + 1))
    return None
def pit():
    """前瞻的點時存檔：申報期間每天抓「正在申報的那一季」，每週一另外檢查最近 4 季有沒有更正；
    內容有變才存 data/fin_pit/YYYYQs_抓取日.csv.gz（保留每個版本、不覆蓋），之後可以知道每家公司第一次出現的日期與後來的更正。"""
    import hashlib
    today = dt.date.today(); os.makedirs("data/fin_pit", exist_ok=True)
    wins = {1: ((4, 20), (5, 25)), 2: ((7, 20), (8, 25)), 3: ((10, 20), (11, 25)), 4: ((2, 15), (4, 10))}
    targets = []
    for s, ((m1, d1), (m2, d2)) in wins.items():
        y = today.year - (1 if s == 4 else 0)
        if dt.date(today.year, m1, d1) <= today <= dt.date(today.year, m2, d2): targets.append((y, s))
    if today.weekday() == 0:
        qi = today.year * 4 + (today.month - 1) // 3 - 1
        targets += [((qi - k) // 4, (qi - k) % 4 + 1) for k in range(4)]
    for y, s in sorted(set(targets)):
        rows = []
        for typ in ("sii", "otc"):
            for api, stmt in (("t163sb04", "is"), ("t163sb05", "bs")):
                tabs = one(api, typ, y, s); time.sleep(4)
                for t_ in tabs or []:
                    t_.columns = [str(c).replace(" ", "") for c in t_.columns]
                    cc = [c for c in t_.columns if "代號" in c][0]; nc = [c for c in t_.columns if "名稱" in c]
                    for _, r in t_.iterrows():
                        code = str(r[cc]).strip()
                        if not code[:4].isdigit(): continue
                        for f_ in t_.columns:
                            if f_ in (cc, *nc): continue
                            v = num(r[f_])
                            if v is not None: rows.append((typ, stmt, code, r[nc[0]] if nc else "", f_, v))
        if not rows: print(y, s, "no data yet"); continue
        df = pd.DataFrame(rows, columns=["market", "stmt", "code", "name", "field", "value"]).sort_values(["market", "stmt", "code", "field"])
        h = hashlib.md5(df.to_csv(index=False).encode()).hexdigest()
        prev = sorted(glob.glob(f"data/fin_pit/{y}Q{s}_*.csv.gz"))
        if prev and hashlib.md5(pd.read_csv(prev[-1], dtype={"code": str}).sort_values(["market", "stmt", "code", "field"]).to_csv(index=False).encode()).hexdigest() == h:
            print(y, s, "unchanged"); continue
        df.to_csv(f"data/fin_pit/{y}Q{s}_{today.isoformat()}.csv.gz", index=False); print(y, s, "saved", df.code.nunique(), "家")

if len(sys.argv) > 1 and sys.argv[1] == "pit":
    import glob
    pit(); sys.exit(0)

today = dt.date.today(); fails = []
for y in range(2013, today.year + 1):
    for s in (1, 2, 3, 4):
        if (y, s) >= (today.year, (today.month - 1) // 3 + 1): continue
        fn = f"data/fin/{y}Q{s}.csv.gz"
        if os.path.exists(fn): continue
        rows = []
        for typ in ("sii", "otc"):
            for api, stmt in (("t163sb04", "is"), ("t163sb05", "bs")):
                tabs = one(api, typ, y, s); time.sleep(4)
                if tabs is None: fails.append(f"{y}Q{s} {typ} {api}"); continue
                for t in tabs:
                    t.columns = [str(c).replace(" ", "") for c in t.columns]
                    cc = [c for c in t.columns if "代號" in c][0]; nc = [c for c in t.columns if "名稱" in c]
                    for _, r in t.iterrows():
                        code = str(r[cc]).strip()
                        if not code[:4].isdigit(): continue
                        for f_ in t.columns:
                            if f_ in (cc, *nc): continue
                            v = num(r[f_])
                            if v is not None: rows.append((typ, stmt, code, r[nc[0]] if nc else "", f_, v))
        if rows:
            pd.DataFrame(rows, columns=["market", "stmt", "code", "name", "field", "value"]).to_csv(fn, index=False)
        print(y, s, len(rows), "rows", flush=True)
open("data/fin/fetch_notes.txt", "w").write("\n".join(fails)); print("failed:", fails or "none")
