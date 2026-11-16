# 強勢股雷達：自動更新

每個平日台北 18:30，GitHub 雲端自動：
1. 抓證交所＋櫃買中心當天收盤行情，存到 `data/prices.csv`
2. 重新計算條件，產生網站 `docs/index.html`
3. 網站 https://dingtingdong.github.io/surge-data/ 自動更新

電腦不用開機。

## 檔案
- `fetch_prices.py`：抓行情
- `build_site.py`：計算條件、產生網站
- `site_template.html`：網站畫面
- `.github/workflows/update.yml`：排程設定

## 更新程式的方法
把新的檔案上傳到 repo 根目錄覆蓋舊檔（Add file → Upload files → Commit），
再到 Actions → update-prices → Run workflow 手動跑一次，網站就會換成新版。

## 排程沒跑時
Actions 裡點紅色那次看錯誤訊息。GitHub 排程常會延遲 10 到 30 分鐘，不一定是壞了。
