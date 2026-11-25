"""每天早上抓期交所台指期（TX）夜盤收盤：寫 docs/night.json（網站上方顯示）、data/night.csv（只增不改），並用 Telegram 通知。
夜盤 15:00～隔天 05:00；期交所把夜盤算在「下一個交易日」。漲跌是相對前一個一般交易時段的收盤。|漲跌| ≥ 2% 特別標出。"""
import csv, datetime as dt, io, json, os, sys
import requests, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import notify as nt
now = dt.datetime.utcnow() + dt.timedelta(hours=8); end = now.date(); start = end - dt.timedelta(days=10)
r = requests.post("https://www.taifex.com.tw/cht/3/futDataDown", data={"down_type": "1", "commodity_id": "TX", "queryStartDate": start.strftime("%Y/%m/%d"), "queryEndDate": end.strftime("%Y/%m/%d")},
                  headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
txt = r.content.decode("cp950", errors="replace")
df = pd.read_csv(io.StringIO(txt), dtype=str); df.columns = [c.strip() for c in df.columns]
print("筆數", len(df), "交易時段", df["交易時段"].astype(str).str.strip().unique()[:5].tolist(), "契約", df["契約"].astype(str).str.strip().unique()[:5].tolist()); print(df.tail(4).to_string())
df = df[(df["契約"].str.strip() == "TX") & ~df["到期月份(週別)"].str.contains("/", na=False)].copy()
for c in ("收盤價", "漲跌價", "成交量"): df[c] = pd.to_numeric(df[c].str.replace(",", "").str.strip(), errors="coerce")
df["交易日期"] = pd.to_datetime(df["交易日期"].str.strip()); df["月"] = df["到期月份(週別)"].str.strip()
night = df[df["交易時段"].str.contains("盤後", na=False)].dropna(subset=["收盤價"])
day = df[df["交易時段"].str.contains("一般", na=False)].dropna(subset=["收盤價"])
if night.empty: print("沒有夜盤資料"); sys.exit(0)
d = night["交易日期"].max(); n = night[night["交易日期"] == d]
n = n.sort_values("成交量", ascending=False).iloc[0]          # 成交量最大的月份＝近月主力
prev = day[(day["交易日期"] < d) & (day["月"] == n["月"])].sort_values("交易日期")
base = float(prev["收盤價"].iloc[-1]) if len(prev) else float(n["收盤價"]) - float(n["漲跌價"])
close = float(n["收盤價"]); chg = close - base; pct = chg / base * 100
rec = {"date": d.strftime("%Y-%m-%d"), "fetched": now.strftime("%Y-%m-%d %H:%M"), "month": n["月"], "close": close, "chg": round(chg, 1), "pct": round(pct, 2), "base": base, "vol": int(n["成交量"] or 0)}
print(rec)
json.dump(rec, open("docs/night.json", "w", encoding="utf-8"), ensure_ascii=False)
newf = not os.path.exists("data/night.csv"); seen = set() if newf else set(pd.read_csv("data/night.csv", dtype=str).date)
if rec["date"] not in seen:
    with open("data/night.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rec)); 
        if newf: w.writeheader()
        w.writerow(rec)
flag = "⚠️ 大於 2%｜" if abs(pct) >= 2 else ""
nt.send(f"night-{rec['date']}", f"【夜盤】{flag}台指期（{rec['month']}）夜盤收 {close:,.0f}，{'+' if chg >= 0 else ''}{chg:,.0f} 點（{'+' if pct >= 0 else ''}{pct:.2f}%），相對前一天日盤收盤 {base:,.0f}。\n夜盤算在 {d:%m/%d} 這個交易日。\nhttps://dingtingdong.github.io/surge-data/")
