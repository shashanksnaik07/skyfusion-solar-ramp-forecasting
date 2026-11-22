# ============================================================
# check_flagged_days_data.py
#
# Purpose: find test days where RMSE > 3, then check whether
# there's an ACTUAL DATA PROBLEM on those days (missing images,
# missing flow, sensor anomalies) — independent of the error
# itself. Uses your already-saved predictions CSV. No retraining,
# no GPU, no TensorFlow needed.
#
# IMPORTANT: this checks the data. It does not decide for you
# whether a day should be excluded. If nothing here comes back
# abnormal, that means the model was wrong on that day — not
# that the data was.
# ============================================================

import os, glob, re
import numpy as np
import pandas as pd
from datetime import datetime

# ── Match these to your training script's config ───────────────
# NOTE: these are the original developer's paths. Replace all four
# with your own exact paths before running.
PRED_CSV     = "/path/to/output/skyfusion_3dcnn_v2_predictions.csv"  # <-- replace with your own predictions CSV path
MERGED_CSV   = "/path/to/POA_Power_cloudiness_aoi_pvwatts_merged_full.csv"  # <-- replace with your own sensor CSV path
IMAGES_ROOT  = "/path/to/sky_images"          # <-- replace with your own images folder path
FLOW_PATH    = "/path/to/flows_precomputed_128_1056.npz"  # <-- replace with your own flow file path
SEQ_LEN      = 7
RMSE_FLAG_THRESHOLD = 3.0
N_BASELINE_DAYS = 8
SEED = 42

# ── Step 1: load predictions, compute per-day RMSE ──────────────
print("[1/4] Loading predictions and computing per-day RMSE...")
pred = pd.read_csv(PRED_CSV, parse_dates=["timestamp"])
pred["day"] = pred["timestamp"].dt.strftime("%Y-%m-%d")

# Use the bias-corrected prediction column if present, else y_pred
pred_col = "y_pred_bc" if "y_pred_bc" in pred.columns else "y_pred"
pred["err_sq"] = (pred["y_true"] - pred[pred_col]) ** 2

per_day = (pred.groupby("day")
           .agg(n=("err_sq", "size"), rmse=("err_sq", lambda x: np.sqrt(x.mean())))
           .reset_index()
           .sort_values("rmse", ascending=False))

flagged = per_day[per_day["rmse"] > RMSE_FLAG_THRESHOLD]
print(f"\nDays with test RMSE > {RMSE_FLAG_THRESHOLD}: {len(flagged)} "
      f"out of {len(per_day)} test days\n")
if len(flagged) > 0:
    print(flagged.to_string(index=False))
else:
    print("None found at this threshold. Nothing to check — stopping here.")
    raise SystemExit(0)

# ── Step 2: build image map and load flow store (cheap, no GPU) ─
print("\n[2/4] Building image map and loading flow data...")

def parse_time(fname):
    m = re.match(r"(\d{14})", os.path.basename(fname))
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None

def floor5(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

img_map = {}
for p in set(glob.glob(os.path.join(IMAGES_ROOT, "**", "*.jpg"), recursive=True)):
    dt = parse_time(p)
    if dt:
        img_map[floor5(dt)] = p
print(f"  Indexed {len(img_map)} images")

flow_npz = np.load(FLOW_PATH)
flow_store = dict(zip(flow_npz["keys"].tolist(), flow_npz["values"]))
print(f"  Loaded {len(flow_store)} precomputed flow pairs")

# ── Step 3: load raw sensor data ────────────────────────────────
print("\n[3/4] Loading raw sensor readings...")
raw = pd.read_csv(MERGED_CSV, sep="\t", parse_dates=["timestamp"])
raw["timestamp"] = raw["timestamp"].dt.tz_localize(None).dt.floor("5min")
raw["day"] = raw["timestamp"].dt.strftime("%Y-%m-%d")

# ── Step 4: check each flagged day's DATA, plus a normal baseline ─
print("\n[4/4] Checking data on flagged days vs a normal-day baseline...\n")

def check_day(d, test_rows_for_day):
    day_raw = raw[raw["day"] == d]
    n_raw = len(day_raw)

    has_image = day_raw["timestamp"].map(lambda t: img_map.get(t) is not None)
    img_coverage = has_image.mean() * 100 if n_raw > 0 else float("nan")

    # Flow coverage: reconstruct each test row's 7-frame window from
    # its own timestamp, exactly like the training script did, then
    # check how many of the 6 frame-to-frame pairs exist in flow_store.
    flow_miss_vals = []
    STEP = pd.Timedelta(minutes=5)
    for t in test_rows_for_day["timestamp"]:
        t_list = [t - (SEQ_LEN - 1 - i) * STEP for i in range(SEQ_LEN)]
        paths = [img_map.get(floor5(tk.to_pydatetime())) for tk in t_list]
        missing = 0
        for k in range(SEQ_LEN - 1):
            if paths[k] is None or paths[k+1] is None:
                missing += 1
                continue
            f1, f2 = os.path.basename(paths[k]), os.path.basename(paths[k+1])
            if f"{f1}__{f2}" not in flow_store:
                missing += 1
        flow_miss_vals.append(missing / (SEQ_LEN - 1))
    flow_missing_pct = np.mean(flow_miss_vals) * 100 if flow_miss_vals else float("nan")

    return {
        "day": d,
        "n_raw_rows": n_raw,
        "n_test_rows": len(test_rows_for_day),
        "img_coverage_pct": round(img_coverage, 1) if n_raw > 0 else None,
        "flow_missing_pct": round(flow_missing_pct, 1),
        "GHI_min": round(day_raw["GHI"].min(), 1) if n_raw > 0 else None,
        "GHI_max": round(day_raw["GHI"].max(), 1) if n_raw > 0 else None,
        "GHI_std": round(day_raw["GHI"].std(), 1) if n_raw > 0 else None,
        "Power_min": round(day_raw["Power"].min(), 2) if n_raw > 0 else None,
        "Power_max": round(day_raw["Power"].max(), 2) if n_raw > 0 else None,
        "Power_std": round(day_raw["Power"].std(), 2) if n_raw > 0 else None,
    }

print("--- Flagged (high-RMSE) days ---")
for d in flagged["day"]:
    test_rows_day = pred[pred["day"] == d]
    print(check_day(d, test_rows_day))

normal_days = per_day[per_day["rmse"] <= RMSE_FLAG_THRESHOLD]["day"]
rng = np.random.RandomState(SEED)
baseline_days = rng.choice(normal_days, size=min(N_BASELINE_DAYS, len(normal_days)),
                            replace=False)

print(f"\n--- Same checks on {len(baseline_days)} normal (non-flagged) "
      f"days, for honest comparison ---")
for d in baseline_days:
    test_rows_day = pred[pred["day"] == d]
    print(check_day(d, test_rows_day))

print("\n" + "="*55)
print("How to read this:")
print("  img_coverage_pct / flow_missing_pct: should look similar on")
print("  flagged and normal days (near 100% / near 0%). A real gap here")
print("  on flagged days = a real data problem, legitimate to flag.")
print("  GHI_std / Power_std near zero = sensor flatlined that day.")
print("  Negative Power, or GHI/Power values way outside the ranges seen")
print("  on normal days, = a real sensor anomaly.")
print("  If none of this looks different between flagged and normal days,")
print("  there is no data problem — the model was simply wrong on those")
print("  days, which is a real result, not something to exclude.")
print("="*55)
