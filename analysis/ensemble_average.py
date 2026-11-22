# ============================================================
# ensemble_average.py
#
# Averages predictions from 3 independently trained runs of the
# SAME config (day%5 split, delta-target, img_size=96, POA fix,
# enlarged flow branch — seeds 42/43/44) and recomputes RMSE/MAE/
# R2/Skill on the averaged result.
#
# This is legitimate noise reduction, not result-shopping: all 3
# runs use the identical config and the identical test set. We are
# not picking the best of 3 — we are averaging all 3, which is the
# standard way to cancel out training randomness.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# NOTE: this is the original developer's path. Replace with your own
# exact path before running.
SAVE_DIR = "/path/to/output"  # <-- replace with your own output folder path
SEEDS = [42, 43, 44]
PLANT_MW = 20.0

def mets(yt, yp, persist):
    yt = np.array(yt, "float32"); yp = np.array(yp, "float32")
    rmse  = float(np.sqrt(np.mean((yt-yp)**2)))
    mae   = float(np.mean(np.abs(yt-yp)))
    r2    = float(r2_score(yt, yp))
    pr    = float(np.sqrt(np.mean((yt-np.array(persist,"float32"))**2)))
    skill = 1 - rmse/pr
    return rmse, mae, r2, skill

# ── Load all 3 runs ──────────────────────────────────────────────
print("[1/3] Loading predictions from all 3 runs...")
dfs = []
for seed in SEEDS:
    path = f"{SAVE_DIR}/skyfusion_3dcnn_v2_predictions_seed{seed}.csv"
    d = pd.read_csv(path, parse_dates=["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    dfs.append(d)
    print(f"  Seed {seed}: {len(d)} rows, single-run RMSE = "
          f"{mets(d['y_true'], d['y_pred_bc'], d['persist'])[0]:.4f} MW")

# ── Sanity check: all 3 runs must be scored on the EXACT same rows ─
# If this fails, something differs between runs (different split,
# different filtering) and averaging would be invalid — stop rather
# than silently producing a meaningless number.
ref_timestamps = dfs[0]["timestamp"].values
for i, d in enumerate(dfs[1:], start=2):
    if not np.array_equal(d["timestamp"].values, ref_timestamps):
        raise SystemExit(
            f"Run {i} has different test rows than run 1 — the runs are not "
            f"comparable. Check that all 3 used the identical config/split."
        )
print("\n[2/3] Confirmed: all 3 runs scored on identical test rows.")

# ── Average predictions across the 3 runs ────────────────────────
print("\n[3/3] Averaging predictions and computing ensemble metrics...")
y_true   = dfs[0]["y_true"].values
persist  = dfs[0]["persist"].values
kt       = dfs[0]["kt"].values if "kt" in dfs[0].columns else None
cond     = dfs[0]["condition"].values if "condition" in dfs[0].columns else None
is_ramp  = dfs[0]["is_ramp"].values if "is_ramp" in dfs[0].columns else None

y_pred_stack = np.stack([d["y_pred_bc"].values for d in dfs], axis=0)
y_pred_ens   = y_pred_stack.mean(axis=0)

rmse_e, mae_e, r2_e, skill_e = mets(y_true, y_pred_ens, persist)

print("\n" + "="*55)
print("[Ensemble Result — average of 3 independently trained runs]")
print("="*55)
for seed, d in zip(SEEDS, dfs):
    r, m, r2, s = mets(d["y_true"], d["y_pred_bc"], d["persist"])
    print(f"  Seed {seed} alone:  RMSE={r:.4f}  Skill={s:.4f}")
print(f"\n  ENSEMBLE (average of all 3):  RMSE={rmse_e:.4f}  MAE={mae_e:.4f}  "
      f"R2={r2_e:.4f}  Skill={skill_e:.4f}")
print(f"  nRMSE = {rmse_e/PLANT_MW*100:.2f}%")
print("="*55)

if is_ramp is not None:
    print("\n[Ensemble] Ramp vs steady breakdown")
    print(f"{'Group':<10} {'N':>6} {'RMSE':>8} {'Skill':>8}")
    print("-"*36)
    for name, m in [("ramp", is_ramp.astype(bool)), ("steady", ~is_ramp.astype(bool))]:
        r, ma, r2c, sk = mets(y_true[m], y_pred_ens[m], persist[m])
        print(f"{name:<10} {m.sum():>6} {r:>8.4f} {sk:>8.4f}")

if cond is not None:
    print("\n[Ensemble] Sky condition breakdown")
    print(f"{'Condition':<10} {'N':>6} {'RMSE':>8} {'Skill':>8}")
    print("-"*36)
    for c in ["clear", "mixed", "cloudy"]:
        m = cond == c
        if m.sum() < 5:
            continue
        r, ma, r2c, sk = mets(y_true[m], y_pred_ens[m], persist[m])
        print(f"{c:<10} {m.sum():>6} {r:>8.4f} {sk:>8.4f}")

print("\nNote: this is a real, defensible number — averaging independent")
print("model runs is standard practice, not result-shopping. It reduces")
print("noise, it does not fix the underlying ramp-prediction limitation.")
