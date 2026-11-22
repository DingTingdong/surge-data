"""臺灣 50 指數換股的事件研究（依 reports/台股強勢股選股擴大研究.md 的 H8；規格照 research/SPEC.md）。

全部在看結果之前寫下（2026/11/22 12:35），看到結果後不修改。
為什麼用「模擬名單」：沒有免費、完整的 2015–2026 歷次成分股異動清單（臺灣指數公司網站是動態載入、維基百科沒有該頁）。
  改用公開規則模擬：每次審核用基準日的市值排名（發行股數 × 收盤價，只含上市普通股），
  非成分股排到第 40 名以內 → 納入；成分股掉到第 61 名以後 → 刪除；再依排名補足或刪減到 50 檔。
  起始名單＝第一次審核基準日市值前 50 名。這份名單在基準日收盤後就能算出，交易上可行、沒有前視。
  限制：沒有套用公眾流通量與流動性篩選、沒有處理臨時異動（下市、合併後的候補遞補），所以會有誤差。
準確度檢查（先做、只報告、不影響判斷）：和新聞報導過的審核比對——2023/12 無增刪；2025/09 納入康霈*(6919)、川湖(2059)，
  刪除藥華藥(6446)、台泥(1101)；2026/03 納入金像電(2368)、鴻勁(7769)、京元電子(2449)、欣興(3037)、華邦電(2344)，
  刪除貿聯-KY(3665)、聯詠(3034)、統一超(2912)、瑞昱(2379)、萬海(2615)；2026/09 納入藥華藥(6446)、刪除世芯-KY(3661)。
時程：審核月 3、6、9、12 月；生效＝該月第 3 個星期五收盤後；公告＝生效前 14 天（星期五）；基準日＝公告前 11 天（星期一）。
  非交易日一律往前找最近的交易日。審核從 2015/09 開始（發行股數資料從 2015/6 起）。
報酬：等權分析範圍（20 日均額 ≥ 1,000 萬）的同窗報酬當基準；扣 0.6% 來回成本；開盤一價漲停買不到不計。
  每次審核的事件先平均成一筆，再對所有審核算 t（審核間隔 3 個月、窗口不重疊）。另報「同一家公司一年只算一次」與相對 0050。
P1（主要）：模擬納入股，從公告隔天開盤買、生效日收盤賣，扣成本後減等權分析範圍：平均 > 0 且 t ≥ 2.9，而且去重版本同方向 → 成立。
P2（主要）：模擬刪除股，從生效隔天開盤買、抱 60 個交易日，扣成本後減等權分析範圍：平均 > 0 且 t ≥ 2.9，而且去重版本同方向 → 成立。
另報（不判定）：納入股 基準日隔天開盤 → 公告日收盤（搶先佈局）、生效後 60 日（是否反轉）；刪除股 公告隔天 → 生效日。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO); sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, "offense.py"), encoding="utf-8").read().split("def build():")[0].split('"""', 2)[2])
KNOWN = {"2023-12": (set(), set()), "2025-09": ({"6919", "2059"}, {"6446", "1101"}),
         "2026-03": ({"2368", "7769", "2449", "3037", "2344"}, {"3665", "3034", "2912", "2379", "2615"}), "2026-09": ({"6446"}, {"3661"})}

def third_friday(y, m):
    d = pd.Timestamp(y, m, 1); d += pd.Timedelta(days=(4 - d.weekday()) % 7); return d + pd.Timedelta(days=14)

def load():
    os.chdir(full_root())
    from radar_core import load_prices, panels, features, tiers
    p = load_prices(); P = panels(p); del p
    f = features(P); S = tiers(f, P["MK"]); os.chdir(REPO)
    C, O, H, L, raw = P["C"], P["O"], P["H"], P["L"], P["rawC"]; days, cols = C.index, C.columns
    X = pd.concat([pd.read_csv(fn, dtype={"code": str}, usecols=["date", "code", "shares"]) for fn in sorted(glob.glob("data/extra/*.csv.gz"))])
    X["date"] = pd.to_datetime(X.date); sh = X.drop_duplicates(["date", "code"], keep="last").pivot(index="date", columns="code", values="shares").reindex(index=days, columns=cols).ffill(limit=5)
    mk = P["MK"].ffill().iloc[-1]
    elig = [c for c in cols if mk.get(c) == "TWSE" and not c.startswith("91")]
    mcap = (sh * raw)[elig]
    inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=cols).fillna(False).astype(bool)
    return C, O, H, L, mcap, inb, days, cols

def simulate(mcap, days):
    td = lambda d: days[days <= d][-1]
    revs = []
    for y in range(2015, 2027):
        for m in (3, 6, 9, 12):
            E = third_friday(y, m); A = E - pd.Timedelta(days=14); K = A - pd.Timedelta(days=11)
            if K < pd.Timestamp("2015-08-01") or E > days[-1]: continue
            revs.append({"rev": f"{y}-{m:02d}", "K": td(K), "A": td(A), "E": td(E)})
    mem = None; out = []
    for r in revs:
        mc = mcap.loc[r["K"]].dropna(); rank = mc.rank(ascending=False, method="first")
        if mem is None: mem = set(rank[rank <= 50].index); continue
        adds = {c for c in rank.index if rank[c] <= 40 and c not in mem}
        dels = {c for c in mem if (c not in rank.index) or rank[c] >= 61}
        new = (mem - dels) | adds
        if len(new) > 50:
            for c in sorted([c for c in new if c not in adds], key=lambda c: -rank.get(c, 1e9))[:len(new) - 50]: new.discard(c)
        if len(new) < 50:
            for c in [c for c in rank.sort_values().index if c not in new][:50 - len(new)]: new.add(c)
        r = dict(r, adds=sorted(new - mem), dels=sorted(mem - new)); out.append(r); mem = new
    return out

if __name__ == "__main__":
    C, O, H, L, mcap, inb, days, cols = load()
    revs = simulate(mcap, days)
    print("== 準確度（和新聞報導比對）")
    for r in revs:
        if r["rev"] in KNOWN:
            ka, kd = KNOWN[r["rev"]]; sa, sd = set(r["adds"]), set(r["dels"])
            print(r["rev"], "基準日", r["K"].date(), "公告", r["A"].date(), "生效", r["E"].date(), "| 納入 對", sorted(sa & ka), "漏", sorted(ka - sa), "多", sorted(sa - ka),
                  "| 刪除 對", sorted(sd & kd), "漏", sorted(kd - sd), "多", sorted(sd - kd))
    print("每次模擬的異動檔數（納入）：", [len(r["adds"]) for r in revs])
    json.dump([{k: (str(v.date()) if hasattr(v, "date") else v) for k, v in r.items()} for r in revs], open(f"{REPO}/research/results/index_sim.json", "w"), ensure_ascii=False, indent=0)
    # ---- 報酬 ----
    pos = {d: i for i, d in enumerate(days)}; N = len(days); Cv, Ov, Hv, Lv = C.values, O.values, H.values, L.values
    prevc = C.ffill().shift(1).values; cj = {c: j for j, c in enumerate(cols)}; Cff = C.ffill(limit=60).values; bidx = bench_index(days).values
    def win(i0, i1, j):  # i0 開盤買、i1 收盤賣
        if i1 >= N or i0 >= N: return None
        o = Ov[i0, j]
        if not o == o or o <= 0 or (o == Hv[i0, j] == Lv[i0, j] and o / prevc[i0, j] - 1 >= LOCK): return None
        return (Cff[i1, j] / o - 1) * 100
    def uni(i0, i1):
        jj = np.where(inb.values[i0 - 1])[0]; o = Ov[i0, jj]; c = Cff[i1, jj]; ok = (o > 0) & np.isfinite(c)
        return float(np.nanmean((c[ok] / o[ok] - 1) * 100))
    rows = []
    for r in revs:
        iK, iA, iE = pos[r["K"]], pos[r["A"]], pos[r["E"]]
        for kind, codes in (("納入", r["adds"]), ("刪除", r["dels"])):
            for c in codes:
                j = cj[c]
                for wname, i0, i1 in (("基準日→公告", iK + 1, iA), ("公告→生效", iA + 1, iE), ("生效後60日", iE + 1, iE + 60)):
                    if i1 >= N: continue
                    y = win(i0, i1, j)
                    if y is None: continue
                    rows.append({"rev": r["rev"], "date": r["A"], "code": c, "kind": kind, "win": wname, "x": y - 0.6 - uni(i0, i1),
                                 "xb": y - 0.6 - (bidx[i1] / bidx[i0 - 1] - 1) * 100})
    T = pd.DataFrame(rows)
    def stat(z):
        s = z.groupby("rev").x.mean(); sb = z.groupby("rev").xb.mean()
        t_ = lambda q: round(float(q.mean() / (q.std(ddof=1) / np.sqrt(len(q)))), 2) if len(q) > 2 else None
        return {"事件數": int(len(z)), "審核次數": int(len(s)), "減等權(每次審核平均)": round(float(s.mean()), 2), "t": t_(s), "減0050": round(float(sb.mean()), 2), "t(0050)": t_(sb),
                "賺贏等權比例%": round(float((z.x > 0).mean() * 100), 1)}
    def dedup(z):
        z = z.sort_values("date"); keep, last = [], {}
        for i, (d, c) in zip(z.index, zip(z.date, z.code)):
            if c in last and (d - last[c]).days < 365: continue
            last[c] = d; keep.append(i)
        return z.loc[keep]
    res = {}
    for kind, wname, lab in (("納入", "公告→生效", "P1 納入股 公告→生效"), ("刪除", "生效後60日", "P2 刪除股 生效後60日"),
                             ("納入", "基準日→公告", "另報 納入股 基準日→公告"), ("納入", "生效後60日", "另報 納入股 生效後60日"), ("刪除", "公告→生效", "另報 刪除股 公告→生效")):
        z = T[(T.kind == kind) & (T.win == wname)]
        res[lab] = {"全部": stat(z), "一年只算一次": stat(dedup(z))}
        print(lab, res[lab], flush=True)
    for k in ("P1 納入股 公告→生效", "P2 刪除股 生效後60日"):
        a_, d_ = res[k]["全部"], res[k]["一年只算一次"]
        res[k]["成立"] = bool(a_["減等權(每次審核平均)"] > 0 and (a_["t"] or 0) >= 2.9 and d_["減等權(每次審核平均)"] > 0)
        print(k, "成立 =", res[k]["成立"])
    json.dump(res, open(f"{REPO}/research/results/index_events.json", "w"), ensure_ascii=False, indent=1)
    T.to_csv(f"{REPO}/research/results/index_events_trades.csv", index=False)
