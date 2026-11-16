"""補 2020 年起的除權息與減資資料（給 maintenance 工作流程用）。"""
import sys, os
sys.path.insert(0, os.getcwd())
import fetch_adjust
fetch_adjust.main("2020-01-01")
