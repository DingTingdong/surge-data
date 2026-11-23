"""存下櫃買中心「轉(交)換債發行資料」（目前流通中的國內可轉債）到 data/cb/issbd5_YYYYMMDD.csv（研究用；每次執行另存一份）。"""
import datetime as dt, os, requests, pandas as pd
j = requests.get("https://www.tpex.org.tw/openapi/v1/bond_ISSBD5_data", headers={"User-Agent": "Mozilla/5.0"}, timeout=60).json()
os.makedirs("data/cb", exist_ok=True); df = pd.DataFrame(j); fn = f"data/cb/issbd5_{(dt.datetime.utcnow() + dt.timedelta(hours=8)):%Y%m%d}.csv"
df.to_csv(fn, index=False); print(fn, df.shape); print(df[["IssuerCode", "IssuerName", "BondCode", "IssueDate", "ListingDate", "IssueAmount", "OutstandingAmount"]].head(5).to_string())
print([c for c in df.columns])
