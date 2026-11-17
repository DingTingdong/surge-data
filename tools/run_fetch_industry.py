import runpy, os, sys
sys.path.insert(0, os.getcwd())
runpy.run_path("fetch_industry.py", run_name="__main__")
