import os, runpy, sys
sys.path.insert(0, os.getcwd()); sys.argv = ["fetch_revenue.py", "2019-01", "2026-10"]
runpy.run_path("fetch_revenue.py", run_name="__main__")
