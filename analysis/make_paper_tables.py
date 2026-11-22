# ============================================================
# make_paper_tables.py
#
# Builds the two tables the paper needs on the FINAL test set:
#   Table 4  baselines vs this work  (RMSE, MAE, R2, nRMSE, FS)
#   Table 5  monthly breakdown
# from the saved Ramp Gate predictions (step2_train_fusion.py,
# FUSION = "ramp_gate"), averaged over the seeds you ran.
#
# No TensorFlow, no GPU, no retraining. Needs only CSV files:
#   - the merged sensor CSV
#   - skyfusion_fusion_ramp_gate_predictions_seed<N>.csv  (one per seed)
#
# Baselines use the paper's definitions (Section 3.2):
#   persistence        P(t+10) = P(t)
#   smart persistence  P(t) * Pcs(t+10)/Pcs(t), ratio clipped to [0.1, 2.0]
#   physical model chain: see the note printed under Table 4
# FS (%) = (1 - RMSE_model / RMSE_smart_persistence) * 100
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# NOTE: these are the original developer's paths. Replace both with your own exact paths.
MERGED_CSV = "/path/to/POA_Power_cloudiness_aoi_pvwatts_merged_full.csv"  # <-- replace with your own sensor CSV path
PRED_DIR   = "/path/to/output"  # <-- replace with the folder holding the ramp_gate prediction CSVs
SEEDS      = [42, 43]
OUT_DIR    = PRED_DIR   # <-- replace if you want the table files somewhere else

DATE_START, DATE_END, GHI_MIN, PLANT_MW = "2024-06-18", "2025-02-16", 10.0, 20.0
SP_CLIP = (0.1, 2.0)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return (float(np.sqrt(np.mean((y - p) ** 2))), float(np.mean(np.abs(y - p))),
            float(r2_score(y, p)))


# ── 1. Rebuild the baselines from the raw sensor CSV ─────────────
print("[1/4] Rebuilding baselines from the sensor CSV...")
df = pd.read_csv(MERGED_CSV, sep="\t", parse_dates=["timestamp"])
df["timestamp"] = df["timestamp"].dt.tz_localize(None)
df = df.sort_values("timestamp").reset_index(drop=True)
df = df[(df["timestamp"] >= pd.Timestamp(DATE_START)) &
        (df["timestamp"] <= pd.Timestamp(DATE_END))].copy()
df = df[df["GHI"] >= GHI_MIN].copy()
df["timestamp"] = df["timestamp"].dt.floor("5min")
df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")
_d = df["day"]
_ip = (_d >= "2024-06-28") & (_d <= "2024-08-30")
_iw = (df["timestamp"].dt.time >= pd.Timestamp("06:30").time()) & \
      (df["timestamp"].dt.time <= pd.Timestamp("09:00").time())
df = df[~(_ip & _iw)].copy()

days = sorted(df["day"].unique())
tr_d = {d for i, d in enumerate(days) if i % 5 < 3}      # train days only -> plant scale factor
df["ghi_cs"] = df["ghi_cs"].replace(0, np.nan).ffill().bfill().fillna(1.)
PLANT_EFF = PLANT_MW / df.loc[df["day"].isin(tr_d), "ghi_cs"].quantile(0.99)
df["P_cs"] = (df["ghi_cs"] * PLANT_EFF).clip(0, PLANT_MW)

# Value 10 minutes ahead, within the same day (no reaching into the next day)
df["P_cs_t10"] = df.groupby("day")["P_cs"].shift(-2)
ratio = (df["P_cs_t10"] / df["P_cs"].clip(lower=1e-6)).clip(*SP_CLIP)
df["power_sp"] = (df["Power"] * ratio).clip(0, PLANT_MW)

if "Pac_modelchain" in df.columns:
    mc = df["Pac_modelchain"].clip(0, PLANT_MW).fillna(0)
    df["mc_now"] = mc
    df["mc_t10"] = mc.groupby(df["day"]).shift(-2)
else:
    df["mc_now"] = df["mc_t10"] = np.nan
    print("  [Warn] Pac_modelchain not in the CSV: model-chain row will be empty.")
lookup = df.set_index("timestamp")[["power_sp", "mc_now", "mc_t10"]]

# ── 2. Load the saved predictions for every seed ─────────────────
print("[2/4] Loading predictions...")
runs = {}
for sd in SEEDS:
    p = os.path.join(PRED_DIR, f"skyfusion_fusion_ramp_gate_predictions_seed{sd}.csv")
    if not os.path.exists(p):
        print(f"  [Skip] {p} not found")
        continue
    d = pd.read_csv(p, parse_dates=["timestamp"])
    d["timestamp"] = d["timestamp"].dt.floor("5min")
    runs[sd] = d.sort_values("timestamp").reset_index(drop=True)
if not runs:
    raise SystemExit("No prediction files found.")
ref = runs[next(iter(runs))]
for sd, d in runs.items():
    if not np.array_equal(d["timestamp"].values, ref["timestamp"].values):
        raise SystemExit(f"Seed {sd} was scored on different test rows; cannot average.")
n_all = len(ref)
print(f"  {len(runs)} seed(s), {n_all} test rows, {ref['timestamp'].dt.strftime('%Y-%m-%d').nunique()} test days")

base = ref[["timestamp", "y_true", "persist"]].join(
    lookup, on="timestamp")                      # baselines aligned to the test rows
# Use rows where the smart-persistence value exists, for ALL methods
valid = base["power_sp"].notna()
print(f"  Rows with a valid smart-persistence value: {valid.sum()} of {n_all} "
      f"(the rest are last-of-day rows with no value 10 min ahead)")

# ── 3. Table 4 ───────────────────────────────────────────────────
print("\n[3/4] Table 4: baselines vs this work")
y = base.loc[valid, "y_true"].values
rows = []
def add_row(name, pred_by_seed):
    vals = np.array([metrics(y, p) for p in pred_by_seed])    # seeds x (rmse, mae, r2)
    rmse, mae, r2 = vals.mean(axis=0)
    rows.append(dict(model=name, rmse=rmse, mae=mae, r2=r2, nrmse=rmse / PLANT_MW * 100))

add_row("Conventional persistence", [base.loc[valid, "persist"].values])
add_row("Smart persistence", [base.loc[valid, "power_sp"].values])
if base["mc_t10"].notna().any():
    v2 = valid & base["mc_t10"].notna()
    if v2.sum() == valid.sum():
        add_row("Physical model chain, P_mc(t+10)", [base.loc[valid, "mc_t10"].values])
    add_row("Physical model chain, P_mc(t)", [base.loc[valid, "mc_now"].values])
add_row("This work", [d.loc[valid.values, "y_pred_bc"].values for d in runs.values()])

t4 = pd.DataFrame(rows)
sp_rmse = t4.loc[t4["model"] == "Smart persistence", "rmse"].iloc[0]
t4["FS_pct"] = (1 - t4["rmse"] / sp_rmse) * 100
print(t4.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
all_rows = np.array([metrics(d["y_true"], d["y_pred_bc"]) for d in runs.values()]).mean(axis=0)
print(f"\nNOTE: Table 4 uses the {valid.sum()} rows where every method has a value. "
      f"On all {n_all} test rows 'This work' scores RMSE {all_rows[0]:.3f}, MAE {all_rows[1]:.3f}, "
      f"R2 {all_rows[2]:.3f} (this is the Overall row of Table 5). Quote one of them as the")
print("headline number and say which rows it covers.")
print("\nNOTE on the model-chain row: this script cannot see how the paper defined the")
print("model-chain forecast. Keep the version that matches your original definition")
print("(either P_mc at the target time or P_mc at the issue time) and delete the other.")

print("\nLaTeX rows for Table 4:")
for r in t4.itertuples():
    print(f"{r.model:<36} & {r.rmse:.3f} & {r.mae:.3f} & {r.r2:.3f} & {r.nrmse:.2f} & "
          f"{r.FS_pct:+.1f} \\\\")

# ── 4. Table 5 (monthly) ─────────────────────────────────────────
print("\n[4/4] Table 5: monthly breakdown (mean over seeds)")
month = ref["timestamp"].dt.strftime("%Y-%m").values
out = []
for ym in sorted(set(month)):
    m = month == ym
    per_seed = np.array([metrics(d.loc[m, "y_true"], d.loc[m, "y_pred_bc"]) for d in runs.values()])
    rmse, mae, r2 = per_seed.mean(axis=0)
    out.append((ym, rmse, mae, r2, int(m.sum())))
per_seed_all = np.array([metrics(d["y_true"], d["y_pred_bc"]) for d in runs.values()]).mean(axis=0)
out.append(("Overall", *per_seed_all, n_all))

t5 = pd.DataFrame(out, columns=["month", "rmse", "mae", "r2", "n"])
print(t5.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nLaTeX rows for Table 5:")
for r in t5.itertuples():
    print(f"{r.month:<8} & {r.rmse:.3f} & {r.mae:.3f} & {r.r2:.3f} & {r.n} \\\\")

t4.to_csv(os.path.join(OUT_DIR, "paper_table4_baselines.csv"), index=False)
t5.to_csv(os.path.join(OUT_DIR, "paper_table5_monthly.csv"), index=False)
print(f"\n[Saved] paper_table4_baselines.csv and paper_table5_monthly.csv in {OUT_DIR}")
