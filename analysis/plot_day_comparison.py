# ============================================================
# plot_day_comparison.py
#
# Plots ONE day: actual power vs smart persistence vs your model's
# prediction, over time. Saves a PNG — no TensorFlow, no GPU.
#
# Ground truth (actual power) is included even though you asked for
# just the other two — without it there's no way to visually judge
# which method is closer. Easy to remove if you really don't want it.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Match these to your training script's config ───────────────
# NOTE: these are the original developer's paths. Replace all three
# with your own exact paths before running.
MERGED_CSV  = "/path/to/POA_Power_cloudiness_aoi_pvwatts_merged_full.csv"  # <-- replace with your own sensor CSV path
PRED_CSV    = "/path/to/output/skyfusion_3dcnn_v2_predictions.csv"  # <-- replace with your own predictions CSV path
OUT_PNG     = "/path/to/output/day_comparison.png"  # <-- replace with your own output image path
DATE_START  = "2024-06-18"
DATE_END    = "2025-02-16"
GHI_MIN     = 10.0
PLANT_MW    = 20.0
P_CS_FLOOR  = 0.05   # numerical safety only
SP_MIN_PCS  = 1.0     # MW — rows below this are excluded (unstable ratio,
                       # confirmed by the sunrise spike you found)

# Which day to plot. Defaulted to the worst-RMSE day found earlier in
# this project (a real ramp day) — change to any "YYYY-MM-DD" you want.
DAY_TO_PLOT = "2024-08-21"

# ── Reload raw data and reproduce the same filters as training ──
print("[1/3] Reloading raw data and computing smart persistence...")
df = pd.read_csv(MERGED_CSV, sep="\t", parse_dates=["timestamp"])
df["timestamp"] = df["timestamp"].dt.tz_localize(None)
df = df.sort_values("timestamp").reset_index(drop=True)
df = df[(df["timestamp"] >= pd.Timestamp(DATE_START)) &
        (df["timestamp"] <= pd.Timestamp(DATE_END))].copy()
df = df[df["GHI"] >= GHI_MIN].copy()
df["timestamp"] = df["timestamp"].dt.floor("5min")
df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")

_ts = pd.to_datetime(df["timestamp"])
_ip = (_ts.dt.strftime("%Y-%m-%d") >= "2024-06-28") & \
      (_ts.dt.strftime("%Y-%m-%d") <= "2024-08-30")
_iw = (_ts.dt.time >= pd.Timestamp("06:30").time()) & \
      (_ts.dt.time <= pd.Timestamp("09:00").time())
df = df[~(_ip & _iw)].copy()

days = sorted(df["day"].unique())
tr_d = {d for i, d in enumerate(days) if i % 5 < 3}

df["ghi_cs"] = df["ghi_cs"].replace(0, np.nan).ffill().bfill().fillna(1.)
train_ghi_cs_q99 = df.loc[df["day"].isin(tr_d), "ghi_cs"].quantile(0.99)
PLANT_EFF = PLANT_MW / train_ghi_cs_q99
df["P_cs"] = (df["ghi_cs"] * PLANT_EFF).clip(0, PLANT_MW).astype(np.float32)
df["P_cs_tplus10"] = df.groupby("day")["P_cs"].shift(-2)
_p_cs_safe = df["P_cs"].clip(lower=P_CS_FLOOR)
df["power_sp"] = (df["Power"] * df["P_cs_tplus10"] / _p_cs_safe).clip(0, PLANT_MW)
df.loc[df["P_cs"] < SP_MIN_PCS, "power_sp"] = np.nan

# ── Join predictions, filter to the one selected day ────────────
print(f"[2/3] Filtering to {DAY_TO_PLOT}...")
pred = pd.read_csv(PRED_CSV, parse_dates=["timestamp"])
pred["timestamp"] = pred["timestamp"].dt.floor("5min")
sp_lookup = df.set_index("timestamp")["power_sp"]
pred["power_sp"] = pred["timestamp"].map(sp_lookup)

pred_col = "y_pred_bc" if "y_pred_bc" in pred.columns else "y_pred"
day_data = pred[pred["timestamp"].dt.strftime("%Y-%m-%d") == DAY_TO_PLOT].copy()
day_data = day_data.sort_values("timestamp")

if len(day_data) == 0:
    available = sorted(pred["timestamp"].dt.strftime("%Y-%m-%d").unique())
    raise SystemExit(
        f"No test rows found for {DAY_TO_PLOT}. This day may not be in "
        f"your TEST split (only ~20% of days are). Days actually available "
        f"in your test set, sample: {available[:10]} ... ({len(available)} total)"
    )
print(f"  Found {len(day_data)} rows for this day")

# ── Plot ──────────────────────────────────────────────────────
print("[3/3] Plotting...")
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(day_data["timestamp"], day_data["y_true"],
        label="Actual power", color="black", linewidth=2)
ax.plot(day_data["timestamp"], day_data[pred_col],
        label="Model prediction", color="tab:blue", linewidth=1.8)
ax.plot(day_data["timestamp"], day_data["power_sp"],
        label="Smart persistence", color="tab:orange", linewidth=1.8,
        linestyle="--")

ax.set_xlabel("Time")
ax.set_ylabel("Power (MW)")
ax.set_title(f"Actual vs Model vs Smart Persistence — {DAY_TO_PLOT}")
ax.legend()
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=130)
print(f"\n[Saved] {OUT_PNG}")

# Quick per-day RMSE for context, so the plot isn't the only evidence
def rmse(a, b):
    a, b = np.array(a, "float32"), np.array(b, "float32")
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < len(a):
        print(f"  (dropped {len(a)-valid.sum()} row(s) with no valid value — "
              f"expected at day boundary, not an error)")
    return float(np.sqrt(np.mean((a[valid]-b[valid])**2)))

print(f"\nOn {DAY_TO_PLOT}:")
model_rmse = rmse(day_data['y_true'], day_data[pred_col])
sp_rmse = rmse(day_data['y_true'], day_data['power_sp'])
print(f"  Model RMSE:             {model_rmse:.4f} MW")
print(f"  Smart persistence RMSE: {sp_rmse:.4f} MW")
