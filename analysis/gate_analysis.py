# ============================================================
# gate_analysis.py
#
# Studies the Ramp Gate values saved by step2_train_fusion.py, for
# Reviewer 1 (point 3) / Reviewer 2 (point 3):
#   - distribution of the gate (does it saturate at 0 or 1, or collapse
#     to one constant value?)
#   - gate during clear / mixed / cloudy conditions
#   - gate during ramp events vs steady periods
#   - correlation with the size of the power change and with cloudiness (kt)
#   - does the gate behave the same across seeds?
#   - gate time series for one day
#   - (if present) the branch weights of the gating_network method
#
# No TensorFlow, no retraining. Reads the prediction CSVs only.
# Needs: skyfusion_fusion_ramp_gate_predictions_seed<N>.csv
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# NOTE: this is the original developer's path. Replace with your own exact path.
SAVE_DIR = "/path/to/output"  # <-- replace with the folder where step2_train_fusion.py saved its CSVs
SEEDS = [42, 43]
DAY = None            # e.g. "2024-08-14". None = the test day with the most ramp rows
SATURATION = 0.05     # gate below 0.05 or above 0.95 counts as saturated
COLLAPSE_STD = 0.02   # gate std below this means it is practically constant

# Colours (validated categorical slots 1-3, light background)
BLUE, ORANGE, AQUA, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "font.size": 10,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True})


def load(path):
    d = pd.read_csv(path, parse_dates=["timestamp"])
    d["change"] = (d["y_true"] - d["persist"]).abs()       # size of the real 10-min change (MW)
    d["day"] = d["timestamp"].dt.strftime("%Y-%m-%d")
    return d


def auc(y, score):
    return float(roc_auc_score(y.astype(int), score)) if y.nunique() > 1 else float("nan")


runs = {}
for sd in SEEDS:
    p = os.path.join(SAVE_DIR, f"skyfusion_fusion_ramp_gate_predictions_seed{sd}.csv")
    if os.path.exists(p):
        d = load(p)
        if "gate" in d.columns:
            runs[sd] = d
        else:
            print(f"[Skip] seed {sd}: file has no 'gate' column")
    else:
        print(f"[Skip] seed {sd}: {p} not found")
if not runs:
    raise SystemExit("No ramp_gate prediction files with gate values found.")

# ── 1. Distribution: saturation / collapse ───────────────────────
print("=" * 78)
print("[1] Gate distribution (test set)")
print("=" * 78)
print(f"{'Seed':>5} {'Mean':>7} {'Std':>7} {'Min':>7} {'P5':>7} {'P50':>7} {'P95':>7} "
      f"{'Max':>7} {'%<0.05':>8} {'%>0.95':>8}")
for sd, d in runs.items():
    g = d["gate"]
    print(f"{sd:>5} {g.mean():>7.3f} {g.std():>7.3f} {g.min():>7.3f} {g.quantile(.05):>7.3f} "
          f"{g.median():>7.3f} {g.quantile(.95):>7.3f} {g.max():>7.3f} "
          f"{(g < SATURATION).mean()*100:>8.1f} {(g > 1-SATURATION).mean()*100:>8.1f}")
    if g.std() < COLLAPSE_STD:
        print(f"      -> COLLAPSED: std {g.std():.4f} < {COLLAPSE_STD}; the gate is practically a constant.")
    elif (g < SATURATION).mean() > .5 or (g > 1 - SATURATION).mean() > .5:
        print("      -> SATURATED: more than half of the rows sit at 0 or 1.")

# ── 2. By sky condition ──────────────────────────────────────────
print("\n" + "=" * 78)
print("[2] Mean gate by sky condition  (kt < 0.4 cloudy, 0.4-0.6 mixed, > 0.6 clear)")
print("=" * 78)
print(f"{'Seed':>5} {'clear':>16} {'mixed':>16} {'cloudy':>16}")
for sd, d in runs.items():
    cells = []
    for c in ["clear", "mixed", "cloudy"]:
        g = d.loc[d["condition"] == c, "gate"]
        cells.append(f"{g.mean():.3f}±{g.std():.3f}(n={len(g)})" if len(g) else "-")
    print(f"{sd:>5} " + " ".join(f"{c:>16}" for c in cells))

# ── 3. Ramp vs steady, correlations, AUC ─────────────────────────
print("\n" + "=" * 78)
print("[3] Gate vs ramp events (top 10% of changes) and vs cloudiness")
print("=" * 78)
print(f"{'Seed':>5} {'Gate|ramp':>10} {'Gate|steady':>12} {'r(|chg|)':>9} {'rho(|chg|)':>11} "
      f"{'r(kt)':>7} {'AUC gate':>9} {'AUC 1-kt':>9}")
rows = []
for sd, d in runs.items():
    isr = d["is_ramp"].astype(bool)
    r_chg = d["gate"].corr(d["change"])
    rho_chg = d["gate"].corr(d["change"], method="spearman")
    r_kt = d["gate"].corr(d["kt"])
    a_gate = auc(isr, d["gate"])
    a_kt = auc(isr, 1 - d["kt"])
    rows.append((sd, r_chg, r_kt, a_gate))
    print(f"{sd:>5} {d.loc[isr,'gate'].mean():>10.3f} {d.loc[~isr,'gate'].mean():>12.3f} "
          f"{r_chg:>9.3f} {rho_chg:>11.3f} {r_kt:>7.3f} {a_gate:>9.3f} {a_kt:>9.3f}")
print("\n  AUC 0.5 = no ability to tell ramp rows from steady rows. 'AUC 1-kt' is the same")
print("  test using only cloudiness, so the gate is only useful beyond cloudiness if its")
print("  AUC is clearly higher. An AUC well below 0.5 means the gate is high on steady")
print("  rows (opposite of the intended meaning).")
signs = {np.sign(r[1]) for r in rows}
if len(rows) > 1 and len(signs) > 1:
    print("  -> The sign of the gate/change correlation DIFFERS between seeds: the gate does")
    print("     not learn a stable meaning.")

# ── 4. Agreement between seeds ───────────────────────────────────
if len(runs) > 1:
    print("\n" + "=" * 78)
    print("[4] Do different seeds give the same gate on the same test rows?")
    print("=" * 78)
    ks = list(runs)
    base = runs[ks[0]].set_index("timestamp")["gate"]
    for sd in ks[1:]:
        other = runs[sd].set_index("timestamp")["gate"]
        j = base.to_frame("a").join(other.to_frame("b"), how="inner")
        print(f"  seed {ks[0]} vs seed {sd}: correlation = {j['a'].corr(j['b']):.3f}  (n={len(j)})")
    print("  Near 1 = same behaviour every run. Near 0 = the gate is mostly random init.")

# ── 5. Gating-network branch weights (if that method was run) ────
gn = {}
for sd in SEEDS:
    p = os.path.join(SAVE_DIR, f"skyfusion_fusion_gating_network_predictions_seed{sd}.csv")
    if os.path.exists(p):
        gn[sd] = load(p)
if gn:
    print("\n" + "=" * 78)
    print("[5] gating_network: mean weight given to each branch (weights sum to 1)")
    print("=" * 78)
    print(f"{'Seed':>5} {'Group':<8} {'RGB':>7} {'Weather':>8} {'Flow':>7}")
    for sd, d in gn.items():
        groups = [("all", np.ones(len(d), bool)), ("clear", d["condition"] == "clear"),
                  ("cloudy", d["condition"] == "cloudy"), ("ramp", d["is_ramp"].astype(bool)),
                  ("steady", ~d["is_ramp"].astype(bool))]
        for name, m in groups:
            if m.sum():
                print(f"{sd:>5} {name:<8} {d.loc[m,'gate_rgb'].mean():>7.3f} "
                      f"{d.loc[m,'gate_wx'].mean():>8.3f} {d.loc[m,'gate_flow'].mean():>7.3f}")

# ── Figures ──────────────────────────────────────────────────────
first = next(iter(runs))
d0 = runs[first]

# Figure 1: distribution, by condition, ramp vs steady (seed = first)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].hist(d0["gate"], bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
ax[0].set_xlim(0, 1); ax[0].set_xlabel("gate value"); ax[0].set_ylabel("test rows")
ax[0].set_title(f"Gate distribution (seed {first})", loc="left", fontsize=10)

conds = ["clear", "mixed", "cloudy"]
data = [d0.loc[d0["condition"] == c, "gate"].values for c in conds]
bp = ax[1].boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                   medianprops=dict(color=INK, linewidth=1.5))
ax[1].set_xticks([1, 2, 3])
ax[1].set_xticklabels([f"{c}\n(n={len(x)})" for c, x in zip(conds, data)])
for patch in bp["boxes"]:
    patch.set_facecolor(BLUE); patch.set_alpha(0.55); patch.set_edgecolor(GREY)
ax[1].set_ylim(0, 1); ax[1].set_ylabel("gate value")
ax[1].set_title("Gate by sky condition", loc="left", fontsize=10)

isr = d0["is_ramp"].astype(bool)
data2 = [d0.loc[~isr, "gate"].values, d0.loc[isr, "gate"].values]
bp2 = ax[2].boxplot(data2, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.5))
ax[2].set_xticks([1, 2])
ax[2].set_xticklabels([f"steady\n(n={len(data2[0])})", f"ramp\n(n={len(data2[1])})"])
for patch, col in zip(bp2["boxes"], [BLUE, ORANGE]):
    patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(GREY)
ax[2].set_ylim(0, 1)
ax[2].set_title("Gate: ramp vs steady rows", loc="left", fontsize=10)
fig.tight_layout()
f1 = os.path.join(SAVE_DIR, "gate_distribution.png")
fig.savefig(f1, dpi=200); plt.close(fig)

# Figure 2: gate vs size of change, per seed (one panel each)
fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4), squeeze=False)
for a, (sd, d) in zip(axes[0], runs.items()):
    a.scatter(d["change"], d["gate"], s=5, color=BLUE, alpha=0.25, linewidths=0)
    a.set_xlabel("size of real 10-minute change (MW)"); a.set_ylabel("gate value")
    a.set_ylim(0, 1)
    a.set_title(f"Seed {sd}: r = {d['gate'].corr(d['change']):+.2f}", loc="left", fontsize=10)
fig.tight_layout()
f2 = os.path.join(SAVE_DIR, "gate_vs_change.png")
fig.savefig(f2, dpi=200); plt.close(fig)

# Figure 3: one-day time series (power on top, gate below, same time axis)
day = DAY or d0.groupby("day")["is_ramp"].sum().idxmax()
g = d0[d0["day"] == day].sort_values("timestamp")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
a1.plot(g["timestamp"], g["y_true"], color=INK, linewidth=1.6, label="actual power")
a1.plot(g["timestamp"], g["y_pred_bc"], color=ORANGE, linewidth=1.4, label="model prediction")
rr = g[g["is_ramp"].astype(bool)]
a1.scatter(rr["timestamp"], rr["y_true"], s=22, color=ORANGE, zorder=3, label="ramp rows")
a1.set_ylabel("power (MW)"); a1.legend(frameon=False, ncol=3, loc="upper left")
a1.set_title(f"{day}: power and Ramp Gate (seed {first})", loc="left", fontsize=10)
a2.plot(g["timestamp"], g["gate"], color=BLUE, linewidth=1.6)
a2.set_ylim(0, 1); a2.set_ylabel("gate value")
import matplotlib.dates as mdates
a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
a2.set_xlabel("time of day")
fig.tight_layout()
f3 = os.path.join(SAVE_DIR, "gate_timeseries.png")
fig.savefig(f3, dpi=200); plt.close(fig)

print("\n[Saved figures]")
for f in (f1, f2, f3):
    print("  ", f)
