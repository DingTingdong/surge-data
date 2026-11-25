"""第 50 輪補充：課程「週線布林上碰上」換不同寫法的敏感度（2026/11/25，使用者問回測是否正確；描述性，不改判定）。2011–2026，20 日減 0050，扣 0.6%。
結果：原本（2σ、4 週內轉向、低於一年高點 85%、週收盤碰上軌）1,839 筆 −3.21（t −4.17）；不要求低檔 7,728 筆 −1.74（t −5.18）；低於 70% −3.95；8 週內轉向 −3.28；
  1.5σ −3.09（t −5.85）；10 週中軸 −3.60（t −7.52）；改用週最高價碰上軌 3,823 筆 −3.18（t −6.94）；週最高價且不要求低檔 9,659 筆 −1.94（t −6.57）。
  老師的例子：國巨 2020-11-20（不要求低檔時抓到；原本因已接近一年高點被排除）、宏達電 2020-12-04（用週最高價碰上軌時抓到；週收盤沒站上上軌）。
  → 每種寫法都明顯為負，結論不變。
"""
import sys,json,numpy as np,pandas as pd,warnings;warnings.filterwarnings("ignore")
sys.path.insert(0,'/home/claude/repo/research')
from vs0050 import nw_t
X=pd.read_pickle('/tmp/model_v2/vs0050_extra.pkl');days=pd.DatetimeIndex(X['days']);N=len(days);cols=list(X['cols']);cj={c:j for j,c in enumerate(cols)}
Cf=pd.DataFrame(X['C'],index=days).ffill(limit=5);Cff=Cf.values;O=np.asarray(X['O'],float);inb=np.asarray(X['inb']).astype(bool);lk=X['lockn'];b=np.asarray(X['bidx'],float)
wk=pd.Series(days.to_period('W-FRI'),index=days);wi=np.where(~wk.duplicated(keep='last').values)[0];W=Cf.iloc[wi]
t0=int(np.searchsorted(days.values,np.datetime64('2011-01-01')))
Hh=pd.DataFrame(np.asarray(X['H'],float),index=days);WH=Hh.groupby(wk.values).max();WH=WH.reindex(pd.PeriodIndex(wk.values[wi]));WH.index=W.index
def run(k=2.0,turnw=4,low=0.85,ma=20,usehigh=False):
    mid=W.rolling(ma,min_periods=ma).mean();sd=W.rolling(ma,min_periods=ma).std();up=mid+k*sd;hi52=W.rolling(52,min_periods=40).max()
    down=np.zeros(mid.shape,bool);md=mid.values
    for q in range(1,turnw+1): down|=np.nan_to_num(np.roll(md,q,0)<np.roll(md,q+1,0)).astype(bool)
    sig=(mid>mid.shift(1)).values&down&((WH if usehigh else W)>=up).values
    if low: sig&=(W<=low*hi52).values
    sig=np.nan_to_num(sig).astype(bool);pr=pd.DataFrame(sig.astype(float)).shift(1).rolling(26,min_periods=1).max().fillna(0).values>0;sig&=~pr
    xs=[];ds=[];hits=[]
    for kk,t in enumerate(wi):
        if t<t0:continue
        for j in np.where(sig[kk]&inb[t])[0]:
            ie=t+1
            if ie+19>=N or not O[ie,j]>0 or lk[ie-1,j]:continue
            ix=ie+19;xs.append((Cff[ix,j]/O[ie,j]-1)*100-0.6-(b[ix]/b[ie-1]-1)*100);ds.append(days[t]);hits.append((cols[j],days[t]))
    s=pd.Series(xs,index=pd.DatetimeIndex(ds));m=s.groupby(s.index.to_period('M')).mean()
    return len(s),round(s.mean(),2),round(s.median(),2),round((s>0).mean()*100),round(nw_t(m.reset_index(drop=True),3),2),hits
for lab,kw in (('週最高價碰上軌（低檔85%）',{'usehigh':True}),('週最高價碰上軌、不要求低檔',{'usehigh':True,'low':None})):
    n,mu,md,w,t,h=run(**kw);print(f'{lab}：{n}筆 平均{mu:+} 中位{md:+} 贏{w}% t{t}')
    print('  宏達電',[str(d.date()) for cc,d in h if cc=='2498' and d.year in (2020,2021)],'國巨',[str(d.date()) for cc,d in h if cc=='2327' and d.year in (2020,2021)])
