"""今日發動的模型特徵（research/normal_model.py 用的同一套 40 個特徵），給訓練（tools/train_model.py）與每天評分（build_site.py）共用。
只用訊號日收盤後就知道的資料。"""
import glob, os
import numpy as np, pandas as pd

def make_context(P, f, S, bear):
    C, O, H, L, A, raw = P["C"], P["O"], P["H"], P["L"], P["A"], P["rawC"]
    days = C.index
    ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
    cap = pd.Series(ind["cap"]).reindex(C.columns); grp = pd.Series(ind["ind_name"]).reindex(C.columns)
    inbase = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
    MA = {n: C.rolling(n, min_periods=n).mean() for n in (5, 20, 60)}
    with np.errstate(divide="ignore", invalid="ignore"):
        pan = {"r60": C / C.shift(60) - 1, "d_hi60": C / C.rolling(60, min_periods=30).max() - 1, "d_hi120": C / C.rolling(120, min_periods=60).max() - 1,
               "m5": C / MA[5] - 1, "m20": C / MA[20] - 1, "m60": C / MA[60] - 1, "nlim60": (f["ch"] >= 9.5).astype(float).shift(1).rolling(60, min_periods=1).sum(),
               "upper": (H - C) / (H - L).replace(0, np.nan), "body": C / O - 1, "gap": O / C.shift(1) - 1, "lprice": np.log(raw), "lmv": np.log(cap * raw / 10),
               "a20": np.log(A.rolling(20, min_periods=10).mean().shift(1))}
    fns = sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))
    ch = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in fns], ignore_index=True)
    ch["date"] = pd.to_datetime(ch.date); ch = ch.drop_duplicates(["date", "code"], keep="last")
    a20 = A.rolling(20, min_periods=10).mean().shift(1)
    for k in ("foreign", "trust", "inst"):
        x = ch.pivot(index="date", columns="code", values=k).reindex(index=days, columns=C.columns) * 1000 * raw
        pan[k + "1"] = x / a20; pan[k + "5"] = x.rolling(5, min_periods=1).sum() / a20
    mg = ch.pivot(index="date", columns="code", values="margin").reindex(index=days, columns=C.columns)
    pan["mg5"] = mg / mg.shift(5) - 1
    rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))], ignore_index=True)
    rv["pub"] = pd.to_datetime(rv.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
    yoy = rv.pivot_table(index="ym", columns="code", values="yoy").sort_index(); pub = rv.groupby("ym").pub.first().sort_index()
    r1 = C.pct_change(fill_method=None).where(inbase)
    gidx = {}
    for g, cs in grp.dropna().groupby(grp.dropna()).groups.items():
        ix = (1 + r1[list(cs)].mean(axis=1).fillna(0)).cumprod(); gidx[g] = (ix / ix.rolling(10).mean() - 1, ix / ix.shift(5) - 1)
    pkd = S[S.pk][["date", "code"]].assign(g=lambda z: z.code.map(grp))
    ewx = (1 + r1.mean(axis=1).fillna(0)).cumprod()
    mkt = pd.DataFrame({"bear": bear.reindex(days), "ew5": ewx / ewx.shift(5) - 1, "ew20": ewx / ewx.shift(20) - 1, "ew_m60": ewx / ewx.rolling(60).mean() - 1,
                        "breadth": (C > MA[20]).where(inbase).mean(axis=1)})
    return {"days": days, "pos": {d: i for i, d in enumerate(days)}, "col": {c: j for j, c in enumerate(C.columns)}, "pan": pan, "yoy": yoy, "pub": pub,
            "grp": grp, "gidx": gidx, "pkd": pkd, "mkt": mkt, "Sx": S.set_index(["date", "code"])}

def features_for(ctx, date, code):
    t, j = ctx["pos"][date], ctx["col"][code]; s = ctx["Sx"].loc[(date, code)]
    feat = {c: s[c] for c in ("ch", "a1", "a20p", "vr", "v", "r5", "r20", "ru", "snp", "rc")}
    feat["a1"] = np.log(max(feat["a1"], 1e-3)); feat["a20p"] = np.log(max(feat["a20p"], 1e-3))
    for kk, pv in ctx["pan"].items(): feat[kk] = pv.iat[t, j]
    yoy, pub = ctx["yoy"], ctx["pub"]; known = pub[pub <= date].index
    feat["yoy1"] = yoy.at[known[-1], code] if len(known) and code in yoy.columns else np.nan
    feat["yoy3"] = yoy.loc[known[-3:], code].mean() if len(known) >= 3 and code in yoy.columns else np.nan
    g = ctx["grp"].get(code); pkd = ctx["pkd"]; days = ctx["days"]
    if isinstance(g, str) and g in ctx["gidx"]:
        feat["g_m10"], feat["g_5"] = ctx["gidx"][g][0].iat[t], ctx["gidx"][g][1].iat[t]
        feat["g_peers"] = int(((pkd.g == g) & (pkd.code != code) & (pkd.date < date) & (pkd.date >= days[max(0, t - 10)])).sum())
    for kk in ctx["mkt"]: feat["mk_" + kk] = ctx["mkt"][kk].iat[t]
    return feat
