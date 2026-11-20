import os, runpy, sys
sys.path.insert(0, os.getcwd()); sys.argv = ["fetch_etf.py", "full"]
runpy.run_path("fetch_etf.py", run_name="__main__")
