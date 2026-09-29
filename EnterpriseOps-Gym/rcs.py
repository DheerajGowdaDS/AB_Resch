import sys, runpy, io, contextlib
sys.argv = ["compute_score.py", "--results_folder", "out/experiment_1/compute_score"]
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        runpy.run_path("compute_score.py", run_name="__main__")
except SystemExit: pass
out = buf.getvalue()
open("cs2.txt","w",encoding="utf-8").write(out)
