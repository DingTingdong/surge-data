"""Telegram 通知（只在 GitHub Actions 有設定 TELEGRAM_BOT_TOKEN、TELEGRAM_CHAT_ID 時才會發；本機什麼都不做）。
同一個 key 只發一次，記在 log 檔（每日更新用 data/notify_log.csv，盤中用 data/notify_log_live.csv，避免兩個流程同時改同一個檔）。
絕對不要印出 token。"""
import csv, datetime as dt, os, requests

def _sent(log):
    if not os.path.exists(log): return set()
    with open(log, encoding="utf-8") as f: return {r["key"] for r in csv.DictReader(f)}

def send(key, text, log="data/notify_log.csv"):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat or key in _sent(log): return False
    try:
        ok = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": text, "disable_web_page_preview": "true"}, timeout=20).status_code == 200
    except Exception:
        ok = False
    if ok:
        new = not os.path.exists(log)
        with open(log, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new: w.writerow(["key", "time"])
            w.writerow([key, (dt.datetime.utcnow() + dt.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")])
    print("notify", key, "sent" if ok else "failed")
    return ok
