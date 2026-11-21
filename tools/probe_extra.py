import os, sys, subprocess
for d in ("2016-01-04", "2026-11-20"): subprocess.run([sys.executable, "tools/fetch_extra.py", "probe", d])
