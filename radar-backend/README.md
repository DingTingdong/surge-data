# Surge Radar 雲端後台

此目錄是新雷達的獨立讀取器，由 GitHub Actions hosted runner 只讀既有 data/prices.csv。它鎖定來源 commit、驗證 Git blob 與原件 SHA、核對最新交易日、股票集合及官方行情，然後執行既有私有 Sites collector 入庫並讀回。執行前會驗證私站程式 manifest 的 SHA-256 與位元組數。不依賴使用者電腦或 ChatGPT，選股公式維持原樣。

`.github/workflows/surge-radar.yml` 是獨立的雷達資料流程，與既有 `update.yml` 的 CSV／靜態網站更新分開執行。

## 執行時段

- 原本 update-prices 成功完成後，自動啟動新雷達讀取器；不改原抓取程式及 18:30 排程。
- 平日台北時間 20:23、21:23、22:23，重試尚未公布的資料。
- 台北時間週二至週六 08:23 執行隔夜補齊檢查。
- Actions 的 `workflow_dispatch` 可立即執行，`require_complete` 預設為 `true`，要求所有來源完成且驗證通過。

GitHub cron 使用 UTC，排程可能延遲；公開 repository 長期沒有活動時，GitHub 也可能停用排程，因此無法保證絕對準點。

CSV 的上櫃量額屬普通交易口徑，新雷達使用完整綜合口徑。讀取器核對股票集合與開高低收，記錄量額差異，寫入時繼續保留原本官方綜合資料；不把對方的選股結果、公式或分數當成本雷達輸入。外資、集保、注意／處置等 CSV 沒有的資料由原官方採集程序補齊。

## 憑證與資料正確性

在 repository Actions secrets 設定 `SURGE_RADAR_SITE_TOKEN`。不要把 token 寫入檔案、日誌或公開 repository。公有 repository 不保存私有雷達引擎、完整採集輸出或憑證；bootstrap 只將必要憑證交給 Sites collector，並只寫允許的摘要到 Actions step summary。

collector 必須檢查官方資料日期與原件雜湊，將資料寫入 D1／R2，並讀回驗證；僅下載成功或程序結束碼為 0 不代表更新完成。未到官方公布時間時，早段排程可以回報資料尚未齊備，但必須清楚標示 `dataComplete` 尚未成立。`workflow_dispatch` 預設要求完整資料；若隔夜補齊檢查仍未取得所有官方來源，工作流程應失敗，讓資料完整性與 workflow 執行健康狀態分開呈現。
