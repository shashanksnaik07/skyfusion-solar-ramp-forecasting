# ============================================================
# compare_runs.py
#
# Compares two saved prediction CSVs (e.g. an old run and the flow-head
# run) to check they were scored on the SAME test rows before comparing
# their RMSE. Then scores both on the rows they share.
#
# No TensorFlow, no retraining.
# ============================================================
import numpy as np
import pandas as pd

# NOTE: these are the original developer's paths. Replace both with your own exact paths.
OLD_CSV = "/path/to/output/skyfusion_3dcnn_v2_predictions_seed42.csv"  # <-- replace with your own OLD run predictions CSV path
NEW_CSV = "/path/to/output/skyfusion_flowhead_predictions_seed42.csv"  # <-- replace with your own NEW run predictions CSV path


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def load(p):
    d = pd.read_csv(p, parse_dates=["timestamp"])
    d["timestamp"] = d["timestamp"].dt.floor("5min")
    d["day"] = d["timestamp"].dt.strftime("%Y-%m-%d")
    return d

old, new = load(OLD_CSV), load(NEW_CSV)
col = lambda d: "y_pred_bc" if "y_pred_bc" in d.columns else "y_pred"

print("=" * 60)
print("[1] Are the test sets the same?")
print("=" * 60)
for name, d in [("OLD", old), ("NEW", new)]:
    pr = rmse(d["y_true"], d["persist"])
    print(f"  {name}: rows={len(d)}  days={d['day'].nunique()}  "
          f"persistence RMSE={pr:.4f}  model RMSE={rmse(d['y_true'], d[col(d)]):.4f}  "
          f"skill={1 - rmse(d['y_true'], d[col(d)]) / pr:.4f}")

only_old = sorted(set(old["day"]) - set(new["day"]))
only_new = sorted(set(new["day"]) - set(old["day"]))
print(f"\n  Days only in OLD test set: {only_old if only_old else 'none'}")
print(f"  Days only in NEW test set: {only_new if only_new else 'none'}")
n_ts_old = len(set(old["timestamp"]) - set(new["timestamp"]))
n_ts_new = len(set(new["timestamp"]) - set(old["timestamp"]))
print(f"  Rows only in OLD: {n_ts_old}   Rows only in NEW: {n_ts_new}")

print("\n" + "=" * 60)
print("[2] Both models scored on the rows they SHARE")
print("=" * 60)
m = old.merge(new, on="timestamp", suffixes=("_old", "_new"))
if len(m) == 0:
    raise SystemExit("No shared rows.")
if not np.allclose(m["y_true_old"], m["y_true_new"], atol=1e-3):
    raise SystemExit("y_true differs on shared timestamps - the two runs use different targets, "
                     "so they cannot be compared row by row.")
yt = m["y_true_old"].values
pers = m["persist_old"].values
pr = rmse(yt, pers)
r_old, r_new = rmse(yt, m[col(old) + "_old"]), rmse(yt, m[col(new) + "_new"])
print(f"  Shared rows: {len(m)}   persistence RMSE = {pr:.4f}")
print(f"  OLD model: RMSE={r_old:.4f}  skill={1 - r_old / pr:.4f}")
print(f"  NEW model: RMSE={r_new:.4f}  skill={1 - r_new / pr:.4f}")
print(f"  Difference (NEW - OLD): {r_new - r_old:+.4f} MW   (noise floor ~0.02)")

print("\n[3] Per-day RMSE of the days that are NOT shared (the ones that moved the score)")
for name, d, days in [("OLD", old, only_old), ("NEW", new, only_new)]:
    for day in days:
        g = d[d["day"] == day]
        print(f"  {name} only  {day}: N={len(g)}  model RMSE={rmse(g['y_true'], g[col(d)]):.3f}")
