"""把 docs/index.html 加密成要輸入密碼才能看的頁面（11/27 使用者要求：不想讓別人看到網站內容）。

密碼放在 GitHub secret SITE_PASSWORD（只有 Actions 讀得到，程式不會印出）。沒有設定就不加密、原樣保留。
做法：PBKDF2-SHA256（31 萬次）由密碼推出金鑰，AES-256-GCM 加密整頁 HTML；瀏覽器用 WebCrypto 解密後顯示。
鹽固定存在 data/site_salt.txt（鹽不是秘密），所以手機記住的金鑰在每天更新後仍然可用；換密碼時刪掉這個檔重產即可。
"""
import base64, os, sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

MARK = "SITE_LOCK_V1"
IT = 310000

def main(path="docs/index.html"):
    pw = os.environ.get("SITE_PASSWORD", "")
    html = open(path, encoding="utf-8").read()
    if MARK in html: print("encrypt: 已經是加密頁，略過"); return
    if not pw: print("encrypt: 沒有設定 SITE_PASSWORD，不加密"); return
    sp = "data/site_salt.txt"
    if not os.path.exists(sp): open(sp, "w").write(base64.b64encode(os.urandom(16)).decode())
    salt = base64.b64decode(open(sp).read().strip())
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=IT).derive(pw.encode("utf-8"))
    iv = os.urandom(12); ct = AESGCM(key).encrypt(iv, html.encode("utf-8"), None)
    b = lambda x: base64.b64encode(x).decode()
    sid = b(salt)[:8]
    page = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="robots" content="noindex,nofollow"><meta name="theme-color" content="#0f1626"><title>強勢股雷達</title>
<link rel="icon" href="favicon.png"><link rel="apple-touch-icon" href="apple-touch-icon.png"><link rel="manifest" href="manifest.webmanifest">
<!-- {MARK} -->
<style>
:root{{color-scheme:light dark;--bg:#f2f4f8;--sf:#fff;--ink:#141a26;--mu:#5d6676;--ln:#dfe3ea;--sig:#b8860b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1626;--sf:#172034;--ink:#e8ecf3;--mu:#8e98aa;--ln:#26314a;--sig:#e0a526}}}}
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);font-family:-apple-system,"Noto Sans TC","PingFang TC",sans-serif}}
form{{width:min(320px,86vw);display:grid;gap:12px}}h1{{font-size:20px;margin:0 0 4px}}
input[type=password]{{width:100%;padding:12px 14px;font-size:17px;border-radius:12px;border:1px solid var(--ln);background:var(--sf);color:var(--ink);outline:none}}
input[type=password]:focus{{border-color:var(--sig)}}
button{{padding:12px;border:0;border-radius:12px;background:var(--sig);color:#111;font-size:16px;font-weight:600}}
label{{font-size:14px;color:var(--mu);display:flex;gap:8px;align-items:center}}label input{{accent-color:var(--sig);width:18px;height:18px}}#e{{color:#d14;font-size:14px;min-height:1em;margin:0}}
</style></head><body>
<form id="f" hidden><h1>強勢股雷達</h1><input id="p" type="password" autocomplete="current-password" placeholder="密碼" autofocus>
<label><input id="r" type="checkbox" checked> 記住這支手機</label><button>進入</button><p id="e"></p></form>
<script>
const S=Uint8Array.from(atob("{b(salt)}"),c=>c.charCodeAt(0)),IV=Uint8Array.from(atob("{b(iv)}"),c=>c.charCodeAt(0)),CT=Uint8Array.from(atob("{b(ct)}"),c=>c.charCodeAt(0)),K="sk_{sid}";
async function show(k){{const pt=await crypto.subtle.decrypt({{name:"AES-GCM",iv:IV}},k,CT);const h=new TextDecoder().decode(pt);if(document.readyState!=="complete")await new Promise(r=>addEventListener("load",r,{{once:true}}));document.open();document.write(h);document.close()}}
async function fromPw(pw){{const b=await crypto.subtle.importKey("raw",new TextEncoder().encode(pw),"PBKDF2",false,["deriveKey"]);return crypto.subtle.deriveKey({{name:"PBKDF2",salt:S,iterations:{IT},hash:"SHA-256"}},b,{{name:"AES-GCM",length:256}},true,["decrypt"])}}
(async()=>{{const f=document.getElementById("f");let s=null;try{{s=localStorage.getItem(K)}}catch(e){{}}
if(s){{try{{const k=await crypto.subtle.importKey("raw",Uint8Array.from(atob(s),c=>c.charCodeAt(0)),"AES-GCM",true,["decrypt"]);await show(k);return}}catch(e){{try{{localStorage.removeItem(K)}}catch(_){{}}}}}}
f.hidden=false;f.onsubmit=async ev=>{{ev.preventDefault();const e=document.getElementById("e");e.textContent="";
try{{const k=await fromPw(document.getElementById("p").value);await crypto.subtle.decrypt({{name:"AES-GCM",iv:IV}},k,CT);
if(document.getElementById("r").checked){{const raw=new Uint8Array(await crypto.subtle.exportKey("raw",k));try{{localStorage.setItem(K,btoa(String.fromCharCode(...raw)))}}catch(_){{}}}}await show(k)}}catch(_){{e.textContent="密碼不對"}}}}}})();
</script></body></html>"""
    open(path, "w", encoding="utf-8").write(page); print("encrypt: 加密成功")

if __name__ == "__main__":
    main(*sys.argv[1:])
