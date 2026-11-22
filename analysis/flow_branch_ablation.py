# ============================================================
# flow_branch_ablation.py
#
# Answers Reviewer 1 (point 2) / Reviewer 2 (point 4):
#   "The Flow branch alone gives RMSE 2.877 MW, but it has no Dense
#    projection layer and no Dropout. Rerun it with both and show
#    the impact."
#
# What this script does
#   Trains the optical-flow branch ALONE (no RGB, no weather) in three
#   versions, each with 2 seeds, under 2 different targets:
#
#   Versions (same conv stack in all three, only the end differs):
#     plain          conv stack -> GAP -> output          (no Dense, no Dropout)
#     dense          conv stack -> GAP -> Dense128 -> Dense64 -> output
#     dense_dropout  conv stack -> GAP -> Dense128 -> Dropout0.2
#                                      -> Dense64  -> Dropout0.1 -> output
#                    (same head as the fused model's head)
#
#   Targets:
#     absolute  predict power 10 min ahead directly (flow only, no
#               knowledge of the current power level).
#     delta     predict the CHANGE in power; absolute power = current
#               power + predicted change (same as the main model).
#
# Why two targets
#   Flow tells you how clouds MOVE. It does not tell you how much
#   power the plant makes right now. If the 2.877 MW result came from
#   the absolute target, the problem may be the missing power level,
#   not the missing Dense/Dropout. Running both targets separates
#   those two explanations. Adding Dense+Dropout cannot fix a missing
#   input.
#
# Same as the main model: day%5 split, rows dropped for missing images
# or flow, sample weights, Huber loss, Adam + cosine restarts, early
# stopping on val loss, hourly bias correction fit on validation.
# So the test rows are identical to step2_train.py's test rows.
#
# Needs: flows_precomputed_128_1056.npz from step1_precompute.py.
# Does NOT load sky images (flow only), so it runs fast.
# Does NOT touch the main model files.
# ============================================================

import os, re, glob, random, time, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
tf.keras.mixed_precision.set_global_policy("float32")


class C:
    # NOTE: these are the original developer's paths. Replace all four with your own exact paths.
    merged_csv  = "/path/to/POA_Power_cloudiness_aoi_pvwatts_merged_full.csv"  # <-- replace with your own sensor CSV path
    images_root = "/path/to/sky_images"  # <-- replace with your own sky-image folder path
    flow_path   = "/path/to/flows_precomputed_128_1056.npz"  # <-- replace with the flow file made by step1_precompute.py
    save_dir    = "/path/to/output"  # <-- replace with your own output folder path

    date_start  = "2024-06-18"
    date_end    = "2025-02-16"
    plant_mw    = 20.0
    ghi_min     = 10.0
    seq_len     = 7
    flow_size   = 48
    # Training (identical to step2_train.py)
    batch       = 16
    epochs      = 100
    patience    = 10
    lr_base     = 1e-4
    huber_delta = 0.1
    ramp_pct    = 0.10
    ramp_w      = 2.0
    cloudy_w    = 2.5
    kt_thr      = 0.40
    # Ablation settings
    seeds       = [42, 43]                      # 2 seeds so the run-to-run noise is visible
    variants    = ["plain", "dense", "dense_dropout"]
    targets     = ["absolute", "delta"]
    noise_floor = 0.02                          # MW; measured run-to-run RMSE noise


flow_ch = 2 * (C.seq_len - 1)


# ============================================================
# Helpers
# ============================================================
def parse_time(fname):
    m = re.match(r"(\d{14})", os.path.basename(fname))
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None

def floor5(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

def build_image_map(root):
    mp = {}
    for p in set(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)):
        dt = parse_time(p)
        if dt:
            mp[floor5(dt)] = p
    return mp

def mets(yt, yp, persist):
    yt, yp = np.array(yt, "float32"), np.array(yp, "float32")
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    r2 = float(r2_score(yt, yp))
    pr = float(np.sqrt(np.mean((yt - np.array(persist, "float32")) ** 2)))
    return rmse, mae, r2, 1 - rmse / pr


# ============================================================
# Data: same rows, same split, same filters as step2_train.py
# ============================================================
def prepare_data():
    print("[Load] Reading CSV...")
    df = pd.read_csv(C.merged_csv, sep="\t", parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Same target definition as step2_train.py (including its known
    # un-grouped shift), so the test rows and targets match exactly.
    if "power_tplus10" not in df.columns:
        df["power_tplus10"] = df["Power"].shift(-2)

    df = df[(df["timestamp"] >= pd.Timestamp(C.date_start)) &
            (df["timestamp"] <= pd.Timestamp(C.date_end))].copy()
    df = df[df["GHI"] >= C.ghi_min].copy()
    df["timestamp"] = df["timestamp"].dt.floor("5min")
    df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    _ts = pd.to_datetime(df["timestamp"])
    _in_period = (_ts.dt.strftime("%Y-%m-%d") >= "2024-06-28") & \
                 (_ts.dt.strftime("%Y-%m-%d") <= "2024-08-30")
    _in_window = (_ts.dt.time >= pd.Timestamp("06:30").time()) & \
                 (_ts.dt.time <= pd.Timestamp("09:00").time())
    df = df[~(_in_period & _in_window)].copy()
    print(f"[Load] {len(df)} rows after filtering")

    days = sorted(df["day"].unique())
    tr_d = {d for i, d in enumerate(days) if i % 5 < 3}
    va_d = {d for i, d in enumerate(days) if i % 5 == 3}
    te_d = {d for i, d in enumerate(days) if i % 5 == 4}

    df["ghi_cs"] = df["ghi_cs"].replace(0, np.nan).ffill().bfill().fillna(1.)
    df["kt"] = (df["GHI"] / df["ghi_cs"]).clip(0, 1.5).fillna(0.)

    img_map = build_image_map(C.images_root)
    flow_npz = np.load(C.flow_path)
    flow_store = dict(zip(flow_npz["keys"].tolist(), flow_npz["values"]))

    def make_index(sub_df):
        STEP = timedelta(minutes=5)
        t = sub_df["timestamp"]
        t_list = [t - (C.seq_len - 1 - i) * STEP for i in range(C.seq_len)]
        ix = pd.DataFrame({f"p{k}": tk.map(img_map.get)
                           for k, tk in enumerate(t_list, start=1)}, index=sub_df.index)
        ix["y_power"] = sub_df["power_tplus10"].astype(float).values
        ix["power_now"] = sub_df["Power"].astype(float).values
        ix["time"] = sub_df["timestamp"].values
        ix["day"] = sub_df["day"].values
        ix["ramp"] = np.abs(ix["y_power"].values - ix["power_now"].values)
        ix["kt"] = sub_df["kt"].clip(0, 1.5).astype(float).values
        ix = ix.dropna(subset=[f"p{k}" for k in range(1, C.seq_len + 1)]).reset_index(drop=True)

        def all_pairs(row):
            for k in range(1, C.seq_len):
                if f"{os.path.basename(str(row[f'p{k}']))}__" \
                   f"{os.path.basename(str(row[f'p{k+1}']))}" not in flow_store:
                    return False
            return True
        ok = ix.apply(all_pairs, axis=1)
        return ix[ok].reset_index(drop=True)

    ix_tr = make_index(df[df["day"].isin(tr_d)])
    ix_va = make_index(df[df["day"].isin(va_d)])
    ix_te = make_index(df[df["day"].isin(te_d)])
    print(f"[Index] Train={len(ix_tr)} Val={len(ix_va)} Test={len(ix_te)} "
          f"(should match step2_train.py's [Index] line)")

    def build_flow_array(ix):
        X = np.zeros((len(ix), C.flow_size, C.flow_size, flow_ch), dtype=np.float16)
        for idx in range(len(ix)):
            row = ix.iloc[idx]
            for k in range(1, C.seq_len):
                key = f"{os.path.basename(str(row[f'p{k}']))}__{os.path.basename(str(row[f'p{k+1}']))}"
                stored = flow_store[key]
                if stored.shape[0] != C.flow_size:
                    stored = cv2.resize(stored.astype(np.float32),
                                        (C.flow_size, C.flow_size)).astype(np.float16)
                X[idx, :, :, (k - 1) * 2:(k - 1) * 2 + 2] = stored
        return X

    print("[Arrays] Building flow arrays...")
    X_tr, X_va, X_te = build_flow_array(ix_tr), build_flow_array(ix_va), build_flow_array(ix_te)

    # Ramp definition: threshold from TRAIN only (same as main script)
    thr = float(np.quantile(ix_tr["ramp"].values, 1 - C.ramp_pct))
    for ix in (ix_tr, ix_va, ix_te):
        ix["is_ramp"] = ix["ramp"].values >= thr

    # Sample weights (train only) - same as main script
    w = np.ones(len(ix_tr), dtype=np.float32)
    w *= np.where(ix_tr["is_ramp"].values, C.ramp_w, 1.)
    w *= np.where(ix_tr["kt"].values < C.kt_thr, C.cloudy_w, 1.)
    months = pd.to_datetime(ix_tr["time"]).dt.month.values
    w *= np.array([{1: 2., 2: 2., 8: 1.5}.get(m, 1.) for m in months], dtype=np.float32)
    w = (w / w.mean()).astype(np.float32)

    return dict(X_tr=X_tr, X_va=X_va, X_te=X_te, ix_tr=ix_tr, ix_va=ix_va, ix_te=ix_te, w=w)


# ============================================================
# Model: flow branch alone, 3 versions
# ============================================================
def build_flow_model(variant, n_train):
    tf.keras.backend.clear_session()
    inp = layers.Input(shape=(C.flow_size, C.flow_size, flow_ch), name="flow")

    # Conv stack: identical to the flow branch in step2_train.py
    f = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    f = layers.BatchNormalization()(f)
    f = layers.MaxPool2D()(f)
    f = layers.Conv2D(64, 3, padding="same", activation="relu")(f)
    f = layers.BatchNormalization()(f)
    f = layers.MaxPool2D()(f)
    f = layers.Conv2D(128, 3, padding="same", activation="relu")(f)
    f = layers.BatchNormalization()(f)
    f = layers.MaxPool2D()(f)
    f = layers.Conv2D(128, 3, padding="same", activation="relu")(f)
    f = layers.BatchNormalization()(f)
    z = layers.GlobalAveragePooling2D()(f)

    if variant == "plain":
        pass                                         # GAP -> output
    elif variant == "dense":
        z = layers.Dense(128, activation="relu")(z)
        z = layers.Dense(64, activation="relu")(z)
    elif variant == "dense_dropout":
        z = layers.Dense(128, activation="relu")(z)
        z = layers.Dropout(0.2)(z)
        z = layers.Dense(64, activation="relu")(z)
        z = layers.Dropout(0.1)(z)
    else:
        raise ValueError(variant)

    out = layers.Dense(1, activation="linear")(z)
    model = Model(inp, out, name=f"flow_only_{variant}")

    steps = max(1, n_train // C.batch)
    sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        C.lr_base, steps * 20, t_mul=1.5, m_mul=0.85, alpha=1e-6)
    model.compile(optimizer=tf.keras.optimizers.Adam(sched, clipnorm=1.0),
                  loss=tf.keras.losses.Huber(delta=C.huber_delta))
    return model


def targets_for(ix, target):
    if target == "delta":
        return ((ix["y_power"].values - ix["power_now"].values) / C.plant_mw).clip(-1, 1).astype("float32")
    return (ix["y_power"].values / C.plant_mw).clip(0, 1).astype("float32")


def to_mw(raw_out, ix, target):
    out = raw_out * C.plant_mw
    if target == "delta":
        out = ix["power_now"].values + out
    return out.clip(0, C.plant_mw)


def run_one(data, variant, target, seed):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    ix_tr, ix_va, ix_te = data["ix_tr"], data["ix_va"], data["ix_te"]
    model = build_flow_model(variant, len(ix_tr))

    t0 = time.time()
    hist = model.fit(
        data["X_tr"], targets_for(ix_tr, target), sample_weight=data["w"],
        validation_data=(data["X_va"], targets_for(ix_va, target)),
        batch_size=C.batch, epochs=C.epochs, shuffle=True, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=C.patience,
                                 restore_best_weights=True)])
    n_ep = len(hist.history["loss"])
    best_ep = int(np.argmin(hist.history["val_loss"])) + 1

    pv = to_mw(model.predict(data["X_va"], batch_size=256, verbose=0).ravel(), ix_va, target)
    pt = to_mw(model.predict(data["X_te"], batch_size=256, verbose=0).ravel(), ix_te, target)
    y_va = ix_va["y_power"].values
    y_te = ix_te["y_power"].values
    persist = ix_te["power_now"].values

    # Hourly bias correction fit on validation (same as main script)
    vh = pd.to_datetime(ix_va["time"]).dt.hour.values
    bh = {h: float((pv[vh == h] - y_va[vh == h]).mean())
          for h in range(24) if (vh == h).sum() > 10}
    th = pd.to_datetime(ix_te["time"]).dt.hour.values
    pt_bc = (pt - np.array([bh.get(h, 0.) for h in th])).clip(0, C.plant_mw)

    rmse_raw = mets(y_te, pt, persist)[0]
    rmse, mae, r2, skill = mets(y_te, pt_bc, persist)
    rm = ix_te["is_ramp"].values
    r_rmse, _, _, r_skill = mets(y_te[rm], pt_bc[rm], persist[rm])
    s_rmse, _, _, s_skill = mets(y_te[~rm], pt_bc[~rm], persist[~rm])

    res = dict(target=target, variant=variant, seed=seed, params=model.count_params(),
               epochs_run=n_ep, best_epoch=best_ep, rmse_raw=rmse_raw, rmse=rmse, mae=mae,
               r2=r2, skill=skill, ramp_rmse=r_rmse, ramp_skill=r_skill,
               steady_rmse=s_rmse, steady_skill=s_skill, seconds=time.time() - t0)
    print(f"  {target:<8} {variant:<14} seed={seed}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"R2={r2:.4f}  skill={skill:+.4f}  ramp_RMSE={r_rmse:.3f}  "
          f"best_ep={best_ep}/{n_ep}  ({res['seconds']:.0f}s)")
    return res


def main():
    os.makedirs(C.save_dir, exist_ok=True)
    data = prepare_data()
    ix_tr, ix_te = data["ix_tr"], data["ix_te"]
    y_te, persist = ix_te["y_power"].values, ix_te["power_now"].values

    # Reference points on the same test rows
    pers_rmse = float(np.sqrt(np.mean((y_te - persist) ** 2)))
    hr_tr = pd.to_datetime(ix_tr["time"]).dt.hour.values
    hr_te = pd.to_datetime(ix_te["time"]).dt.hour.values
    clim = {h: ix_tr["y_power"].values[hr_tr == h].mean() for h in np.unique(hr_tr)}
    clim_pred = np.array([clim.get(h, ix_tr["y_power"].mean()) for h in hr_te])
    clim_rmse = float(np.sqrt(np.mean((y_te - clim_pred) ** 2)))
    print("\n[Reference on the same test rows]")
    print(f"  Persistence RMSE                       = {pers_rmse:.4f} MW")
    print(f"  Hour-of-day average power (no images)  = {clim_rmse:.4f} MW")
    print("  (A model that only knows the time of day scores the second number.)")

    print(f"\n[Run] {len(C.targets)} targets x {len(C.variants)} variants x "
          f"{len(C.seeds)} seeds = {len(C.targets)*len(C.variants)*len(C.seeds)} runs\n")
    rows = []
    for target in C.targets:
        for variant in C.variants:
            for seed in C.seeds:
                rows.append(run_one(data, variant, target, seed))

    res = pd.DataFrame(rows)
    out_csv = os.path.join(C.save_dir, "flow_branch_ablation_results.csv")
    res.to_csv(out_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"[Summary] mean over seeds {C.seeds}   (noise floor ~{C.noise_floor} MW RMSE)")
    print("=" * 78)
    print(f"{'Target':<9} {'Variant':<14} {'Params':>9} {'RMSE':>14} {'MAE':>7} "
          f"{'Skill':>8} {'RampRMSE':>9}")
    agg = {}
    for (target, variant), g in res.groupby(["target", "variant"], sort=False):
        agg[(target, variant)] = g["rmse"].mean()
        print(f"{target:<9} {variant:<14} {int(g['params'].iloc[0]):>9,} "
              f"{g['rmse'].mean():>8.4f}±{g['rmse'].std(ddof=0):<5.3f} "
              f"{g['mae'].mean():>7.4f} {g['skill'].mean():>+8.4f} {g['ramp_rmse'].mean():>9.3f}")

    print("\n[Effect of adding Dense + Dropout]  (plain -> dense_dropout)")
    for target in C.targets:
        d = agg[(target, "dense_dropout")] - agg[(target, "plain")]
        verdict = ("within noise, no real effect" if abs(d) < C.noise_floor
                   else ("real improvement" if d < 0 else "real degradation"))
        print(f"  {target:<9}: {agg[(target,'plain')]:.4f} -> {agg[(target,'dense_dropout')]:.4f} MW "
              f"(change {d:+.4f})  => {verdict}")
        d2 = agg[(target, "dense_dropout")] - agg[(target, "dense")]
        print(f"  {'':<9}  Dropout alone (dense -> dense_dropout): change {d2:+.4f}")

    if "absolute" in C.targets and "delta" in C.targets:
        a = agg[("absolute", "dense_dropout")]; dl = agg[("delta", "dense_dropout")]
        print(f"\n[Effect of giving the model the current power level]  (best variant)")
        print(f"  absolute target : {a:.4f} MW")
        print(f"  delta target    : {dl:.4f} MW   (persistence = {pers_rmse:.4f})")
        print("  If this gap is much larger than the Dense+Dropout effect above, the")
        print("  flow-only weakness comes from the missing power level, not from")
        print("  missing layers.")

    print(f"\n[Saved] {out_csv}")


if __name__ == "__main__":
    main()
