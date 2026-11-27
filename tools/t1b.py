"""今日分數 T1-B 前瞻模擬帳戶（11/27 第十四份覆核：不能只存排名，要固定規則、每天照做、存下每一筆）。

規則（同 research/today_v3.py T1-B，固定不改）：
  資金分 10 批，第 k 批在起始日後第 k 個交易日第一次到期，之後每 10 個交易日到期一次。
  到期那天開盤：持股「前一個交易日的正式今日分數（融資到齊版）仍在前 5 名」就續抱，其他賣出；空出的名額（每批 5 格）
  依前一天前 5 名的排名補進「其他批沒有持有」的股票，開盤比前一天收盤漲 ≥ 9.5% 就不買換下一檔；其餘資金放 0050。
  停牌（沒有開盤價）的要賣的股票，之後第一個有開盤價的交易日開盤賣。
  成本：個股買、賣各 0.3%（來回 0.6%）；0050 只對淨買賣扣 0.09%。除權息用 adjust.csv 的比例把股數放大（等於股利再投入）。
  起始：2026-11-30 開盤全部買 0050，1.0 單位。
紀錄：data/t1b_state.json（目前狀態）、data/t1b_log.csv（每天淨值與交易，只增不改）。
"""
import json, os
import numpy as np, pandas as pd
START = "2026-11-30"; STATE = "data/t1b_state.json"; LOG = "data/t1b_log.csv"; FEE = 0.003; ECOST = 0.0009; H = 10; K = 5

def _ranks(date):
    """前一天的正式排名（mg_ok=1）；沒有正式版就用暫定版並標記。"""
    fn = f"data/tscore/{date[:4]}.csv.gz"
    if not os.path.exists(fn): return None, None
    t = pd.read_csv(fn, dtype={"code": str}, usecols=["date", "code", "rank", "mg_ok"]); t = t[t.date == date]
    if not len(t): return None, None
    ok = int(t.mg_ok.max()); t = t[t.mg_ok == ok]; return t.set_index("code")["rank"].sort_values(), ok

def run(P, days):
    raw = P["rawC"]; rawO = P["O"] * (P["rawC"] / P["C"])
    e = pd.read_csv("data/etf.csv", dtype={"code": str}); e = e[e.code == "0050"].assign(date=lambda z: pd.to_datetime(z.date)).drop_duplicates("date", keep="last").set_index("date").sort_index()
    adj = pd.read_csv("data/adjust.csv", dtype={"code": str}); adj["date"] = pd.to_datetime(adj.date)
    adj = adj[(adj.ratio > 0.05) & (adj.ratio < 20)].groupby(["date", "code"]).ratio.prod()
    ds = [d for d in days if d >= pd.Timestamp(START)]
    if not ds: return {"start": START, "n": 0}
    st = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else None
    log = pd.read_csv(LOG, dtype={"date": str}) if os.path.exists(LOG) else None
    if st is None:
        d0 = ds[0]; px = float(e.open.get(d0, np.nan))
        if not px == px: return {"start": START, "n": 0}
        st = {"last": None, "tr": [{"due": k + 1, "etf": (1.0 / H) * (1 - ECOST) / px, "pos": []} for k in range(H)], "pend": [], "day": 0}
        st["d0"] = d0.strftime("%Y-%m-%d")
    rows = []
    for d in ds:
        dstr = d.strftime("%Y-%m-%d")
        if st["last"] is not None and dstr <= st["last"]: continue
        i = days.get_loc(d); prev = days[i - 1].strftime("%Y-%m-%d")
        # 除權息：股數放大（ratio＝參考價÷前收，<1）
        for q in st["tr"]:
            r50 = adj.get((d, "0050"), 1.0); q["etf"] /= r50
            for p in q["pos"]: p["sh"] /= adj.get((d, p["c"]), 1.0)
        op = lambda c: float(rawO.at[d, c]) if c in rawO.columns and rawO.at[d, c] == rawO.at[d, c] and rawO.at[d, c] > 0 else None
        cl = lambda c: float(raw[c].loc[:d].dropna().iloc[-1]) if c in raw.columns and raw[c].loc[:d].notna().any() else 0.0
        eo, ec = float(e.open.get(d, np.nan)), float(e.close.get(d, np.nan))
        trades = []
        # 停牌延後要賣的
        for q in st["tr"]:
            keep = []
            for p in q["pos"]:
                if p.get("sell") and op(p["c"]):
                    v = p["sh"] * op(p["c"]) * (1 - FEE); q["etf"] += v * (1 - ECOST) / eo; trades.append(f"賣{p['c']}")
                else: keep.append(p)
            q["pos"] = keep
        st["day"] = st.get("day", 0) + (0 if st["last"] is None else 1)
        rk, ok = _ranks(prev)
        top = list(rk[rk <= K].index) if rk is not None else []
        for q in st["tr"]:
            if q["due"] != st["day"]: continue
            q["due"] += H
            if rk is None: continue   # 沒有前一天排名（例如資料缺）：這批維持原狀
            sold = 0.0; keep = []
            for p in q["pos"]:
                if p["c"] in top: keep.append(p); continue
                o = op(p["c"])
                if o is None: p["sell"] = True; keep.append(p); continue
                sold += p["sh"] * o * (1 - FEE); trades.append(f"賣{p['c']}")
            q["pos"] = keep
            held = {p["c"] for x in st["tr"] for p in x["pos"]}
            ev = q["etf"] * eo; tot = sold + ev + sum(p["sh"] * (op(p["c"]) or cl(p["c"])) for p in keep); each = tot / K
            avail = sold + ev * (1 - ECOST); spent = 0.0
            for c in top:
                if len(q["pos"]) >= K: break
                if c in held: continue
                o = op(c); pc = float(raw[c].iloc[i - 1]) if c in raw.columns else np.nan
                if o is None or (pc == pc and pc > 0 and o / pc - 1 >= 0.095 - 1e-9): continue
                amt = min(each, avail - spent)
                if amt <= 0: break
                q["pos"].append({"c": c, "sh": amt * (1 - FEE) / o, "buy": dstr}); spent += amt; held.add(c); trades.append(f"買{c}")
            net = sold - spent
            q["etf"] = (ev + (net * (1 - ECOST) if net >= 0 else net / (1 - ECOST))) / eo
        sv = sum(p["sh"] * cl(p["c"]) for q in st["tr"] for p in q["pos"]); etv = sum(q["etf"] for q in st["tr"]) * ec
        rows.append({"date": dstr, "nav": round(sv + etv, 6), "stock": round(sv / (sv + etv), 4) if sv + etv > 0 else 0, "n": sum(len(q["pos"]) for q in st["tr"]),
                     "rank_ok": ok, "trades": " ".join(trades), "made": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")})
        st["last"] = dstr
    if rows:
        new = pd.DataFrame(rows); (pd.concat([log, new], ignore_index=True) if log is not None else new).to_csv(LOG, index=False)
        json.dump(st, open(STATE, "w", encoding="utf-8"), ensure_ascii=False)
    L = pd.read_csv(LOG, dtype={"date": str}) if os.path.exists(LOG) else None
    if L is None or not len(L): return {"start": START, "n": 0}
    d0 = pd.Timestamp(st["d0"]); b0 = float(e.open.get(d0, np.nan)); bl = float(e.close.loc[:days[-1]].iloc[-1])
    f50 = float(np.prod([adj.get((x, "0050"), 1.0) for x in days if d0 < x <= days[-1]]))
    hold = sorted({p["c"] for q in st["tr"] for p in q["pos"]})
    return {"start": st["d0"], "n": len(L), "nav": round(float(L.nav.iloc[-1]), 4), "b": round(bl / f50 / b0 * (1 - ECOST), 4), "stock": float(L.stock.iloc[-1]), "hold": hold}
