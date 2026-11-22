# ============================================================
# smart_persistence.py
#
# Computes the smart persistence baseline:
#   Power_sp(t+dt) = Power(t) * P_cs(t+dt) / P_cs(t)
#
# and compares it against naive persistence and your model's
# predictions, on the same test rows. No TensorFlow, no GPU,
# no retraining — reads your already-saved predictions CSV and
# the raw merged CSV directly.
#
# This reproduces the SAME filtering, day%5 split, and P_cs
# calculation as the training script, independently, so the
# comparison is apples-to-apples with what your model was
# actually scored against.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ── Match these to your training script's config ───────────────
# NOTE: these are the original developer's paths. Replace both with
# your own exact paths before running.
MERGED_CSV  = "/path/to/POA_Power_cloudiness_aoi_pvwatts_merged_full.csv"  # <-- replace with your own sensor CSV path
PRED_CSV    = "/path/to/output/skyfusion_3dcnn_v2_predictions.csv"  # <-- replace with your own predictions CSV path
DATE_START  = "2024-06-18"
DATE_END    = "2025-02-16"
GHI_MIN     = 10.0
PLANT_MW    = 20.0
P_CS_FLOOR  = 0.05   # MW — numerical safety only, prevents literal divide-by-zero
SP_MIN_PCS  = 1.0     # MW — 5% of plant capacity. Rows where P_cs(t) is below
                       # this are EXCLUDED from the comparison entirely (not just
                       # floored) — near sunrise/sunset the ratio explodes even
                       # with the floor in place, since P_cs(t+10) can be many
                       # times P_cs(t) while both are still small in absolute
                       # terms. Confirmed visually: see day_comparison.png.

def mets(yt, yp, persist):
    yt = np.array(yt, "float32"); yp = np.array(yp, "float32")
    rmse  = float(np.sqrt(np.mean((yt-yp)**2)))
    mae   = float(np.mean(np.abs(yt-yp)))
    r2    = float(r2_score(yt, yp))
    pr    = float(np.sqrt(np.mean((yt-np.array(persist,"float32"))**2)))
    skill = 1 - rmse/pr
    return rmse, mae, r2, skill

# ── Step 1: reload and filter the raw data, same as training ───
print("[1/4] Reloading raw data and reproducing the training filters...")
df = pd.read_csv(MERGED_CSV, sep="\t", parse_dates=["timestamp"])
df["timestamp"] = df["timestamp"].dt.tz_localize(None)
df = df.sort_values("timestamp").reset_index(drop=True)

df = df[(df["timestamp"] >= pd.Timestamp(DATE_START)) &
        (df["timestamp"] <= pd.Timestamp(DATE_END))].copy()
df = df[df["GHI"] >= GHI_MIN].copy()
df["timestamp"] = df["timestamp"].dt.floor("5min")
df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")

# Same morning-anomaly exclusion window as the training script
_ts = pd.to_datetime(df["timestamp"])
_ip = (_ts.dt.strftime("%Y-%m-%d") >= "2024-06-28") & \
      (_ts.dt.strftime("%Y-%m-%d") <= "2024-08-30")
_iw = (_ts.dt.time >= pd.Timestamp("06:30").time()) & \
      (_ts.dt.time <= pd.Timestamp("09:00").time())
df = df[~(_ip & _iw)].copy()
print(f"  {len(df)} rows after filtering (should match your training log)")

# ── Step 2: reproduce the day%5 split, just to get train days ──
# Only needed to derive PLANT_EFF the same way training did — fit
# on train days only, no leakage.
days = sorted(df["day"].unique())
tr_d = {d for i, d in enumerate(days) if i % 5 < 3}

# ── Step 3: clean ghi_cs and compute P_cs, same formula as training ─
print("[2/4] Computing P_cs (clear-sky power)...")
df["ghi_cs"] = df["ghi_cs"].replace(0, np.nan).ffill().bfill().fillna(1.)

train_ghi_cs_q99 = df.loc[df["day"].isin(tr_d), "ghi_cs"].quantile(0.99)
PLANT_EFF = PLANT_MW / train_ghi_cs_q99
df["P_cs"] = (df["ghi_cs"] * PLANT_EFF).clip(0, PLANT_MW).astype(np.float32)

# ── Step 4: smart persistence ────────────────────────────────────
# Day-grouped shift — does NOT reach across the night into the next
# day's clear-sky value (unlike the main power_tplus10 target, which
# uses a global shift — a separate issue, flagged elsewhere).
print("[3/4] Computing smart persistence...")
df["P_cs_tplus10"] = df.groupby("day")["P_cs"].shift(-2)
_p_cs_safe = df["P_cs"].clip(lower=P_CS_FLOOR)
df["power_sp"] = (df["Power"] * df["P_cs_tplus10"] / _p_cs_safe).clip(0, PLANT_MW)
df.loc[df["P_cs"] < SP_MIN_PCS, "power_sp"] = np.nan

# ── Step 5: join onto your saved test predictions by timestamp ──
print("[4/4] Joining onto saved predictions and computing comparison...")
pred = pd.read_csv(PRED_CSV, parse_dates=["timestamp"])
pred["timestamp"] = pred["timestamp"].dt.floor("5min")

sp_lookup = df.set_index("timestamp")["power_sp"]
pred["power_sp"] = pred["timestamp"].map(sp_lookup)

pred_col = "y_pred_bc" if "y_pred_bc" in pred.columns else "y_pred"
valid = pred["power_sp"].notna()
n_dropped = (~valid).sum()

y_true   = pred.loc[valid, "y_true"].values
persist  = pred.loc[valid, "persist"].values
y_pred   = pred.loc[valid, pred_col].values
sp       = pred.loc[valid, "power_sp"].values

rmse_naive,  mae_naive,  r2_naive,  _          = mets(y_true, persist, persist)
rmse_sp,     mae_sp,     r2_sp,     skill_sp   = mets(y_true, sp,      persist)
rmse_model,  mae_model,  r2_model,  skill_model = mets(y_true, y_pred, persist)
skill_model_vs_sp = 1 - rmse_model / rmse_sp

print("\n" + "="*55)
print("[Baseline Comparison] Naive persistence vs Smart persistence vs Model")
print("="*55)
if n_dropped > 0:
    print(f"  Note: {n_dropped} test rows excluded from smart persistence")
    print(f"  (day-boundary rows where P_cs(t+10) isn't available, plus")
    print(f"  sunrise/sunset rows where P_cs(t) < {SP_MIN_PCS} MW — the ratio")
    print(f"  is unstable there, confirmed by the spike in day_comparison.png).")
    print(f"  Excluded from all 3 columns below, so the comparison is on the")
    print(f"  same {valid.sum()} rows for all three methods.")
print(f"\n{'Method':<20} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'Skill*':>8}")
print("-"*54)
print(f"{'Naive persistence':<20} {rmse_naive:>8.4f} {mae_naive:>8.4f} {r2_naive:>8.4f} {'—':>8}")
print(f"{'Smart persistence':<20} {rmse_sp:>8.4f} {mae_sp:>8.4f} {r2_sp:>8.4f} {skill_sp:>8.4f}")
print(f"{'Model':<20} {rmse_model:>8.4f} {mae_model:>8.4f} {r2_model:>8.4f} {skill_model:>8.4f}")
print("\n  *Skill = improvement over NAIVE persistence (standard definition).")
print("  Smart persistence's own skill shows how much it beats naive")
print("  persistence on its own — a real baseline, not a trivial one.")
print(f"\n  Model vs smart persistence directly: "
      f"{rmse_model:.4f} vs {rmse_sp:.4f} MW "
      f"({'model wins' if rmse_model < rmse_sp else 'smart persistence wins'})")
print("="*55)
