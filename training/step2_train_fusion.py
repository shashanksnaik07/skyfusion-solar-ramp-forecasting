# ============================================================
# step2_train_fusion.py
# FUSION COMPARISON for the SkyFusion model (Reviewer 1 pt 3 / Reviewer 2 pt 3).
#
# Trains the SAME model with five different ways of combining the image,
# weather and flow branches, and compares them:
#   ramp_gate, concat, add, attention, gating_network
# Data, split, features, sample weights, loss, schedule, branches and
# prediction head are identical to step2_train.py. Only the fusion step
# differs. For the two methods with a gate, the gate values on the test
# set are saved so analysis/gate_analysis.py can study them.
#
# Run time: about 40 minutes per run on a GPU. 5 methods x 2 seeds = 10
# runs. Finished runs are skipped if you stop and restart.
# Edit FUSIONS / SEEDS near the bottom of this file.
#
# Based on step2_train.py:
# SkyFusion 3D-CNN — 10-minute-ahead solar power ramp forecasting
#
# Inputs per prediction: a 30-minute sequence of 7 sky-camera images
# (5 min apart), the matching optical-flow motion field between
# frames, and 33 weather/irradiance/PV features. Target: the CHANGE
# in power over the next 10 minutes (not the absolute power level).
# ============================================================

import gc, os, sys, re, glob, random, warnings, time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Normalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import pvlib
from pvlib.location import Location

warnings.filterwarnings("ignore")
tf.keras.backend.clear_session()
tf.keras.mixed_precision.set_global_policy("float32")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"[GPU] Using {gpus[0].name}")
    except Exception:
        pass


# ============================================================
# Config
# ============================================================
class C:
    # NOTE: these are the original developer's paths. Replace all four with your own exact paths.
    merged_csv  = "/path/to/POA_Power_cloudiness_aoi_pvwatts_merged_full.csv"  # <-- replace with your own sensor CSV path
    images_root = "/path/to/sky_images"  # <-- replace with your own sky-image folder path
    flow_path   = "/path/to/flows_precomputed_128_1056.npz"  # <-- replace with the flow file made by step1_precompute.py
    save_dir    = "/path/to/output"  # <-- replace with your own output folder path
    date_start  = "2024-06-18"
    date_end    = "2025-02-16"
    plant_mw    = 20.0
    ghi_min     = 10.0          # rows below this GHI are treated as night
    # Site coordinates (Copper Crossing, San Tan Valley, AZ)
    lat         = 33.22
    lon         = -111.55
    alt         = 471.0
    tz          = "Etc/GMT+7"
    # Image / sequence config
    seq_len     = 7              # 30 min of history (7 frames, 5 min apart)
    img_size    = 96              # RGB crop resolution fed to the 3D-CNN
    cx, cy, r   = 0.5, 0.5, 0.46   # crop center (fraction of image) + radius
    flow_size   = 48
    # Training
    batch       = 16
    epochs      = 100
    patience    = 10
    lr_base     = 1e-4
    huber_delta = 0.1
    seed        = 42
    # Sample weights (upweight hard cases in the training loss only)
    ramp_pct    = 0.10   # top 10% of |actual change| counts as a "ramp"
    ramp_w      = 2.0
    cloudy_w    = 2.5
    kt_thr      = 0.40

os.makedirs(C.save_dir, exist_ok=True)
site = Location(C.lat, C.lon, tz=C.tz, altitude=C.alt, name="Copper Crossing")


# ============================================================
# Weather / irradiance / PV feature list (33 total)
# ============================================================
WX_COLS = [
    # Time encoding (4)
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    # Sky image (1)
    "Cloudiness",
    # Solar irradiance (17)
    "GHI", "ghi_cs", "kt", "POA", "POA_cs",
    "ghi_lag5", "ghi_lag10", "ghi_roll15", "ghi_roll30",
    "R_ghi_5", "R_ghi_30",
    "poa_lag5", "poa_lag10", "poa_roll15", "poa_roll30",
    "R_poa_5", "R_poa_30",
    # Weather (4)
    "AirTemp", "cell_temp", "Wind speed", "WindDir",
    # PV generation (7)
    "Power", "power_lag5", "power_lag10",
    "R_p_5", "R_p_30", "P_cs", "P_mc",
]
print(f"[Config] {len(WX_COLS)} weather features")


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

def load_crop(path, S=C.img_size):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((S, S, 3), dtype=np.float32)
    h, w = img.shape[:2]
    cx, cy = int(w * C.cx), int(h * C.cy)
    r = int(min(h, w) * C.r)
    crop = img[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)]
    crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

def mets(yt, yp, persist):
    """RMSE, MAE, R2, and Skill (improvement over a persistence baseline)."""
    yt, yp = np.array(yt, "float32"), np.array(yp, "float32")
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    r2 = float(r2_score(yt, yp))
    pr = float(np.sqrt(np.mean((yt - np.array(persist, "float32")) ** 2)))
    skill = 1 - rmse / pr
    return rmse, mae, r2, skill


# ============================================================
# Load and filter data
# ============================================================
print("[Load] Reading CSV...")
df = pd.read_csv(C.merged_csv, sep="\t", parse_dates=["timestamp"])
df["timestamp"] = df["timestamp"].dt.tz_localize(None)
df = df.sort_values("timestamp").reset_index(drop=True)

# Regression target: power 10 minutes ahead (2 rows, at 5-min resolution).
# NOTE (known limitation): this shift is NOT grouped by day. On the last
# 1-2 daylight rows of each day, it can reach past the removed nighttime
# gap into the next day's early rows rather than returning "no target."
# Affects a small fraction of rows, concentrated at dusk. Not fixed here
# to keep this version's results comparable to everything reported so
# far — flag as a known limitation if writing this up.
if "power_tplus10" not in df.columns:
    df["power_tplus10"] = df["Power"].shift(-2)

df = df[(df["timestamp"] >= pd.Timestamp(C.date_start)) &
        (df["timestamp"] <= pd.Timestamp(C.date_end))].copy()
df = df[df["GHI"] >= C.ghi_min].copy()
df["timestamp"] = df["timestamp"].dt.floor("5min")
df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")

# Remove a known bad-data window (sensor/camera issue during this range)
_ts = pd.to_datetime(df["timestamp"])
_in_period = (_ts.dt.strftime("%Y-%m-%d") >= "2024-06-28") & \
             (_ts.dt.strftime("%Y-%m-%d") <= "2024-08-30")
_in_window = (_ts.dt.time >= pd.Timestamp("06:30").time()) & \
             (_ts.dt.time <= pd.Timestamp("09:00").time())
df = df[~(_in_period & _in_window)].copy()
print(f"[Load] {len(df)} rows after filtering")

# ── Train/val/test split ────────────────────────────────────
# Day-level, 1-in-5 interleaved (every 5th day -> test, every 5th -> val).
# Computed on whole days BEFORE feature engineering, so train-only
# statistics (like the clear-sky power normalisation below) can be
# derived without leaking test-set information into them.
days = sorted(df["day"].unique())
tr_d = {d for i, d in enumerate(days) if i % 5 < 3}
va_d = {d for i, d in enumerate(days) if i % 5 == 3}
te_d = {d for i, d in enumerate(days) if i % 5 == 4}


# ============================================================
# Feature engineering
# ============================================================
print("[Features] Computing weather/irradiance/PV features...")

# Time encoding
dt_ts = pd.to_datetime(df["timestamp"])
hr = (dt_ts.dt.hour + dt_ts.dt.minute / 60) / 24 * 2 * np.pi
doy_period = dt_ts.dt.dayofyear.apply(
    lambda d: 366 if dt_ts.dt.year.max() == 2024 else 365)
dy = (dt_ts.dt.dayofyear - 1) / doy_period * 2 * np.pi
df["hour_sin"], df["hour_cos"] = np.sin(hr).astype(np.float32), np.cos(hr).astype(np.float32)
df["doy_sin"], df["doy_cos"] = np.sin(dy).astype(np.float32), np.cos(dy).astype(np.float32)

# Clear-sky GHI and clearness index
df["ghi_cs"] = df["ghi_cs"].replace(0, np.nan).ffill().bfill().fillna(1.)
df["kt"] = (df["GHI"] / df["ghi_cs"]).clip(0, 1.5).fillna(0.)

# ── Plane-of-array irradiance (tracker-aware, via pvfactors) ────
print("[POA] Computing tracker-aware POA...")
try:
    from pvlib.bifacial.pvfactors import pvfactors_timeseries

    timestamps = pd.DatetimeIndex(df["timestamp"])
    solar_pos = site.get_solarposition(timestamps)
    tracker_data = pvlib.tracking.singleaxis(
        apparent_zenith=solar_pos["apparent_zenith"],
        apparent_azimuth=solar_pos["azimuth"],
        axis_tilt=0, axis_azimuth=180, max_angle=45,
        backtrack=True, gcr=0.35,
    )
    cs = site.get_clearsky(timestamps, model="ineichen")

    disc_out = pvlib.irradiance.boland(
        df["GHI"].values, solar_pos["zenith"].values, timestamps)
    dni_meas = pd.Series(disc_out["dni"], index=df.index).clip(0).fillna(0)
    dhi_meas = pd.Series(disc_out["dhi"], index=df.index).clip(0).fillna(0)

    poa_result = pvfactors_timeseries(
        solar_azimuth=solar_pos["azimuth"].values,
        solar_zenith=solar_pos["apparent_zenith"].values,
        surface_azimuth=tracker_data["surface_azimuth"].fillna(180).values,
        surface_tilt=tracker_data["surface_tilt"].fillna(0).values,
        axis_azimuth=0, timestamps=timestamps,
        dni=dni_meas.values, dhi=dhi_meas.values,
        gcr=0.35, pvrow_height=1.5, pvrow_width=2.0, albedo=0.25,
        n_pvrows=3, index_observed_pvrow=1,
    )
    df["POA"] = np.array(poa_result[0]).clip(0).astype(np.float32)

    poa_cs_result = pvfactors_timeseries(
        solar_azimuth=solar_pos["azimuth"].values,
        solar_zenith=solar_pos["apparent_zenith"].values,
        surface_azimuth=tracker_data["surface_azimuth"].fillna(180).values,
        surface_tilt=tracker_data["surface_tilt"].fillna(0).values,
        axis_azimuth=0, timestamps=timestamps,
        dni=cs["dni"].values, dhi=cs["dhi"].values,
        gcr=0.35, pvrow_height=1.5, pvrow_width=2.0, albedo=0.25,
        n_pvrows=3, index_observed_pvrow=1,
    )
    df["POA_cs"] = np.array(poa_cs_result[0]).clip(0).astype(np.float32)
    print("[POA] pvfactors computed successfully")

except Exception as e:
    print(f"[POA] pvfactors unavailable ({e}), falling back to GHI-based estimate")
    if "effective_POA" in df.columns:
        df["POA"] = df["effective_POA"].clip(0).fillna(0).astype(np.float32)
        df["POA_cs"] = df.get("poa_cs", df["ghi_cs"] * 1.1).clip(0).fillna(0).astype(np.float32)
    else:
        df["POA"] = df["GHI"].clip(0).astype(np.float32)
        df["POA_cs"] = df["ghi_cs"].clip(0).astype(np.float32)

# ── Lag / rolling / ramp-rate features, computed PER DAY ────────
# Grouped by day so a "5-minute lag" never reaches into the previous
# day's last reading (which would otherwise happen silently, since
# night rows are filtered out and days sit back-to-back in the data).
def day_shift(s, day, periods):
    return s.groupby(day).transform(lambda x: x.shift(periods).bfill())

def day_roll_mean(s, day, window):
    return s.groupby(day).transform(lambda x: x.rolling(window, min_periods=1).mean())

def day_ramp(s, day, periods, minutes):
    return s.groupby(day).transform(lambda x: (x - x.shift(periods)).fillna(0) / minutes)

_day = df["day"]
df["ghi_lag5"]   = day_shift(df["GHI"], _day, 1)
df["ghi_lag10"]  = day_shift(df["GHI"], _day, 2)
df["ghi_roll15"] = day_roll_mean(df["GHI"], _day, 3)
df["ghi_roll30"] = day_roll_mean(df["GHI"], _day, 6)
df["R_ghi_5"]    = day_ramp(df["GHI"], _day, 1, 5.0)
df["R_ghi_30"]   = day_ramp(df["GHI"], _day, 6, 30.0)

df["poa_lag5"]   = day_shift(df["POA"], _day, 1)
df["poa_lag10"]  = day_shift(df["POA"], _day, 2)
df["poa_roll15"] = day_roll_mean(df["POA"], _day, 3)
df["poa_roll30"] = day_roll_mean(df["POA"], _day, 6)
df["R_poa_5"]    = day_ramp(df["POA"], _day, 1, 5.0)
df["R_poa_30"]   = day_ramp(df["POA"], _day, 6, 30.0)

df["power_lag5"]  = day_shift(df["Power"], _day, 1)
df["power_lag10"] = day_shift(df["Power"], _day, 2)
df["R_p_5"]       = day_ramp(df["Power"], _day, 1, 5.0)
df["R_p_30"]      = day_ramp(df["Power"], _day, 6, 30.0)

# ── Physics-based clear-sky power ───────────────────────────────
# Normalisation constant fit on TRAIN DAYS ONLY, then applied to all
# rows — avoids leaking test-set irradiance statistics into a feature
# the model trains on.
train_ghi_cs_q99 = df.loc[df["day"].isin(tr_d), "ghi_cs"].quantile(0.99)
PLANT_EFF = C.plant_mw / train_ghi_cs_q99
df["P_cs"] = (df["ghi_cs"] * PLANT_EFF).clip(0, C.plant_mw).astype(np.float32)

if "Pac_modelchain" in df.columns:
    df["P_mc"] = df["Pac_modelchain"].clip(0, C.plant_mw).fillna(0).astype(np.float32)
elif "Pac_cs" in df.columns:
    df["P_mc"] = df["Pac_cs"].clip(0, C.plant_mw).fillna(0).astype(np.float32)
else:
    df["P_mc"] = df["P_cs"]
    print("[Warning] Pac_modelchain not found, using P_cs as P_mc fallback")

for col in WX_COLS:
    if col in df.columns and df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
    elif col not in df.columns:
        print(f"[Warning] Feature '{col}' not found in DataFrame")

WX_COLS = [c for c in WX_COLS if c in df.columns]
print(f"[Features] Final count: {len(WX_COLS)} features")


# ============================================================
# Apply the split
# ============================================================
df_tr = df[df["day"].isin(tr_d)].copy()
df_va = df[df["day"].isin(va_d)].copy()
df_te = df[df["day"].isin(te_d)].copy()
print(f"[Split] Train={len(df_tr)} Val={len(df_va)} Test={len(df_te)}")

print("[Maps] Building image map...")
img_map = build_image_map(C.images_root)
flow_npz = np.load(C.flow_path)
flow_store = dict(zip(flow_npz["keys"].tolist(), flow_npz["values"]))
flow_ch = 2 * (C.seq_len - 1)


# ============================================================
# Build per-row image-sequence index
# ============================================================
def make_index(sub_df):
    STEP = timedelta(minutes=5)
    t = sub_df["timestamp"]
    t_list = [t - (C.seq_len - 1 - i) * STEP for i in range(C.seq_len)]
    path_cols = {f"p{k}": tk.map(img_map.get) for k, tk in enumerate(t_list, start=1)}
    ix = pd.DataFrame(path_cols, index=sub_df.index)

    ix["y_power"] = sub_df["power_tplus10"].astype(float).values
    ix["power_now"] = sub_df["Power"].astype(float).values

    # Target = CHANGE in power (normalised), not the absolute level.
    # "No change" is then output 0, so the model doesn't need to
    # reconstruct the whole diurnal power curve on every steady-state
    # row — only the deviation from persistence.
    ix["y_delta_mw"] = ix["y_power"].values - ix["power_now"].values
    ix["y_norm"] = (ix["y_delta_mw"].values / C.plant_mw).clip(-1, 1)

    ix["wx"] = list(sub_df.loc[:, WX_COLS].astype(float).values)
    ix["time"] = sub_df["timestamp"].values
    ix["day"] = sub_df["day"].values
    ix["ramp"] = np.abs(ix["y_power"].values - ix["power_now"].values)
    ix["kt"] = sub_df["kt"].clip(0, 1.5).astype(float).values

    # Drop rows missing any of the 7 required images
    ix = ix.dropna(subset=[f"p{k}" for k in range(1, C.seq_len + 1)]).reset_index(drop=True)

    # Drop rows where any of the 6 frame-to-frame flow pairs is missing
    # from the precomputed flow file — e.g. images added to the folder
    # AFTER step1_precompute.py ran, so no flow was ever computed for
    # them. Previously this silently zero-filled that part of the flow
    # input and continued; now the row is excluded instead, so results
    # never rest on blank motion data without you knowing about it.
    def all_flow_pairs_present(row):
        for k in range(1, C.seq_len):
            f1 = os.path.basename(str(row[f"p{k}"]))
            f2 = os.path.basename(str(row[f"p{k+1}"]))
            if f"{f1}__{f2}" not in flow_store:
                return False
        return True

    has_flow = ix.apply(all_flow_pairs_present, axis=1)
    n_dropped = int((~has_flow).sum())
    if n_dropped > 0:
        print(f"  [Flow check] Dropping {n_dropped} row(s) with incomplete "
              f"optical-flow coverage (likely images added after "
              f"step1_precompute.py last ran — re-run it to recover these rows)")
    return ix[has_flow].reset_index(drop=True)

ix_tr = make_index(df_tr)
ix_va = make_index(df_va)
ix_te = make_index(df_te)
print(f"[Index] Train={len(ix_tr)} Val={len(ix_va)} Test={len(ix_te)}")


# ============================================================
# Build weather and flow arrays
# ============================================================
def build_wx_arrays(ix_tr, ix_va, ix_te):
    X_tr_raw = np.stack(ix_tr["wx"].to_list(), axis=0).astype(np.float32)
    X_va_raw = np.stack(ix_va["wx"].to_list(), axis=0).astype(np.float32)
    X_te_raw = np.stack(ix_te["wx"].to_list(), axis=0).astype(np.float32)
    with tf.device('/CPU:0'):
        norm = Normalization()
        norm.adapt(X_tr_raw)  # fit on train only
        X_tr = np.nan_to_num(norm(X_tr_raw).numpy(), nan=0., posinf=0., neginf=0.)
        X_va = np.nan_to_num(norm(X_va_raw).numpy(), nan=0., posinf=0., neginf=0.)
        X_te = np.nan_to_num(norm(X_te_raw).numpy(), nan=0., posinf=0., neginf=0.)
    return X_tr, X_va, X_te, norm

print("[Arrays] Normalising weather features...")
X_wx_tr, X_wx_va, X_wx_te, norm_wx = build_wx_arrays(ix_tr, ix_va, ix_te)

def build_flow_array(ix):
    N = len(ix)
    X = np.zeros((N, C.flow_size, C.flow_size, flow_ch), dtype=np.float16)
    for idx in range(N):
        row = ix.iloc[idx]
        for k in range(1, C.seq_len):
            f1 = os.path.basename(str(row[f"p{k}"]))
            f2 = os.path.basename(str(row[f"p{k+1}"]))
            key = f"{f1}__{f2}"
            if key in flow_store:
                ch = (k - 1) * 2
                stored = flow_store[key]
                if stored.shape[0] != C.flow_size:
                    stored = cv2.resize(stored.astype(np.float32),
                                         (C.flow_size, C.flow_size)).astype(np.float16)
                X[idx, :, :, ch:ch+2] = stored
    return X

print("[Arrays] Building flow arrays...")
X_flow_tr = build_flow_array(ix_tr)
X_flow_va = build_flow_array(ix_va)
X_flow_te = build_flow_array(ix_te)


# ============================================================
# Sample weights (train set only)
# ============================================================
# One fixed definition of "ramp" (top 10% by change, threshold set from
# TRAIN data only), applied consistently to train/val/test everywhere
# in this script.
ramp_thr = float(np.quantile(ix_tr["ramp"].values, 1 - C.ramp_pct))
ix_tr["is_ramp"] = ix_tr["ramp"].values >= ramp_thr
ix_va["is_ramp"] = ix_va["ramp"].values >= ramp_thr
ix_te["is_ramp"] = ix_te["ramp"].values >= ramp_thr

w = np.ones(len(ix_tr), dtype=np.float32)
w *= np.where(ix_tr["is_ramp"].values, C.ramp_w, 1.)
w *= np.where(ix_tr["kt"].values < C.kt_thr, C.cloudy_w, 1.)
tr_months = pd.to_datetime(ix_tr["time"]).dt.month.values
w *= np.array([{1: 2., 2: 2., 8: 1.5}.get(m, 1.) for m in tr_months], dtype=np.float32)
w = (w / w.mean()).astype(np.float32)


# ============================================================
# tf.data pipeline
# ============================================================
def load_frames(pjt, S=C.img_size, seq=C.seq_len):
    j = pjt.numpy() if hasattr(pjt, "numpy") else pjt
    if isinstance(j, bytes):
        j = j.decode("utf-8")
    paths = j.split("|")
    frames = []
    for p in paths:
        if p and p != "nan" and os.path.exists(p):
            frames.append(load_crop(p, S).astype(np.float32))
        else:
            frames.append(np.zeros((S, S, 3), dtype=np.float32))
    return np.stack(frames, axis=0)

def make_dataset(ix, X_wx, X_flow, y, weights=None, shuffle=False, seed=42):
    paths_list = [[str(ix.iloc[i][f"p{k}"]) for k in range(1, C.seq_len + 1)]
                  for i in range(len(ix))]
    paths_ds = tf.data.Dataset.from_tensor_slices(["|".join(p) for p in paths_list])
    wx_ds = tf.data.Dataset.from_tensor_slices(X_wx.astype(np.float32))
    flow_ds = tf.data.Dataset.from_tensor_slices(X_flow.astype(np.float32))
    y_ds = tf.data.Dataset.from_tensor_slices(y.astype(np.float32))

    if weights is not None:
        full_ds = tf.data.Dataset.zip((paths_ds, wx_ds, flow_ds, y_ds,
                  tf.data.Dataset.from_tensor_slices(weights.astype(np.float32))))
    else:
        full_ds = tf.data.Dataset.zip((paths_ds, wx_ds, flow_ds, y_ds))
    if shuffle:
        full_ds = full_ds.shuffle(2000, seed=seed)

    def lrgb(pj):
        rgb = tf.py_function(func=load_frames, inp=[pj], Tout=tf.float32)
        rgb.set_shape((C.seq_len, C.img_size, C.img_size, 3))
        return rgb

    if weights is not None:
        ds = full_ds.map(lambda pj, wx, fl, lb, wt: (
            {"rgb": lrgb(pj), "weather": wx, "flow": fl}, lb, wt),
            num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = full_ds.map(lambda pj, wx, fl, lb: (
            {"rgb": lrgb(pj), "weather": wx, "flow": fl}, lb),
            num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(C.batch).prefetch(tf.data.AUTOTUNE)

y_tr = ix_tr["y_norm"].astype(np.float32).values
y_va = ix_va["y_norm"].astype(np.float32).values
y_te = ix_te["y_norm"].astype(np.float32).values

ds_va = make_dataset(ix_va, X_wx_va, X_flow_va, y_va)
ds_te = make_dataset(ix_te, X_wx_te, X_flow_te, y_te)


# ============================================================
# Experiment settings  (EDIT THESE)
# ============================================================
# Fusion methods to compare. Everything before the fusion step (the three
# branches) and everything after it (the prediction head) is identical.
#   ramp_gate       the paper's method: one scalar gate scales RGB and flow features
#   concat          concatenate the three branch features, no gate
#   add             project each branch to 128 values and add them (fixed addition)
#   attention       self-attention across the three branch features (4 heads)
#   gating_network  softmax gate over the 3 branches (a learned weight per branch)
FUSIONS = ["ramp_gate", "concat", "add", "attention", "gating_network"]
SEEDS = [42, 43]          # all methods are run for seed 42 first, then seed 43
SAVE_MODELS = False       # True also saves a .keras file per run (about 6 files per seed)


# ============================================================
# Model — 3 branches (image / weather / flow), then ONE fusion step
# ============================================================
def proj(x, name):
    """Project a branch to 128 values so branches can be added/attended."""
    return layers.Dense(128, activation="relu", name=name)(x)

def as_token(x, name):
    return layers.Reshape((1, 128), name=name)(x)


def build_model(fusion, wx_dim, n_train, seed):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed); random.seed(seed); np.random.seed(seed)
    S = C.img_size

    # Image branch: 3D-CNN over the 7-frame sequence
    rgb_in = layers.Input(shape=(C.seq_len, S, S, 3), name="rgb")
    x = layers.Conv3D(16, 3, padding="same", activation="relu", name="c3d_1")(rgb_in)
    x = layers.BatchNormalization(name="bn3d_1")(x)
    x = layers.MaxPool3D((1, 2, 2), name="mp3d_1")(x)
    x = layers.Conv3D(32, 3, padding="same", activation="relu", name="c3d_2")(x)
    x = layers.BatchNormalization(name="bn3d_2")(x)
    x = layers.MaxPool3D((1, 2, 2), name="mp3d_2")(x)
    x = layers.Conv3D(64, 3, padding="same", activation="relu", name="c3d_3")(x)
    x = layers.BatchNormalization(name="bn3d_3")(x)
    x = layers.MaxPool3D((2, 2, 2), name="mp3d_3")(x)
    x = layers.Conv3D(128, 3, padding="same", activation="relu", name="c3d_4")(x)
    x = layers.BatchNormalization(name="bn3d_4")(x)
    rgb_feat = layers.GlobalAveragePooling3D(name="gap3d")(x)
    rgb_feat = layers.Dense(128, activation="relu", name="rgb_d1")(rgb_feat)
    rgb_feat = layers.Dropout(0.2, name="rgb_drop")(rgb_feat)

    # Weather branch: MLP over the engineered features
    wx_in = layers.Input(shape=(wx_dim,), name="weather")
    w1 = layers.Dense(128, activation="relu", name="wx_d1")(wx_in)
    w1 = layers.Dropout(0.1, name="wx_drop1")(w1)
    w2 = layers.Dense(128, activation="relu", name="wx_d2")(w1)
    w_res = layers.Add(name="wx_res")([w1, w2])
    wx_f = layers.Dense(64, activation="relu", name="wx_d3")(w_res)

    # Flow branch: 2D-CNN over the optical-flow field
    flow_in = layers.Input(shape=(C.flow_size, C.flow_size, flow_ch), name="flow")
    f = layers.Conv2D(32, 3, padding="same", activation="relu", name="fl_c1")(flow_in)
    f = layers.BatchNormalization(name="fl_bn1")(f)
    f = layers.MaxPool2D(name="fl_p1")(f)
    f = layers.Conv2D(64, 3, padding="same", activation="relu", name="fl_c2")(f)
    f = layers.BatchNormalization(name="fl_bn2")(f)
    f = layers.MaxPool2D(name="fl_p2")(f)
    f = layers.Conv2D(128, 3, padding="same", activation="relu", name="fl_c3")(f)
    f = layers.BatchNormalization(name="fl_bn3")(f)
    f = layers.MaxPool2D(name="fl_p3")(f)
    f = layers.Conv2D(128, 3, padding="same", activation="relu", name="fl_c4")(f)
    f = layers.BatchNormalization(name="fl_bn4")(f)
    fl_f = layers.GlobalAveragePooling2D(name="fl_gap")(f)

    # ── Fusion step: the ONLY part that differs between methods ──
    if fusion == "ramp_gate":
        # One scalar gate in [0,1] from all three branches; it scales the
        # RGB and flow features. Gated and ungated copies are both kept.
        gi = layers.Concatenate(name="gate_in")([rgb_feat, wx_f, fl_f])
        gh = layers.Dense(64, activation="relu", name="gate_h1")(gi)
        gh = layers.Dropout(0.1, name="gate_drop")(gh)
        go = layers.Dense(1, activation="sigmoid", name="ramp_gate")(gh)
        rgb_gated = layers.Multiply(name="rgb_gated")([rgb_feat, go])
        flow_gated = layers.Multiply(name="flow_gated")([fl_f, go])
        fuse = layers.Concatenate(name="fuse")([rgb_feat, rgb_gated, wx_f, fl_f, flow_gated])

    elif fusion == "concat":
        fuse = layers.Concatenate(name="fuse")([rgb_feat, wx_f, fl_f])

    elif fusion == "add":
        p_r, p_w, p_f = proj(rgb_feat, "proj_rgb"), proj(wx_f, "proj_wx"), proj(fl_f, "proj_flow")
        fuse = layers.Add(name="fuse")([p_r, p_w, p_f])

    elif fusion == "attention":
        p_r, p_w, p_f = proj(rgb_feat, "proj_rgb"), proj(wx_f, "proj_wx"), proj(fl_f, "proj_flow")
        tok = layers.Concatenate(axis=1, name="tokens")(
            [as_token(p_r, "tok_rgb"), as_token(p_w, "tok_wx"), as_token(p_f, "tok_flow")])
        att = layers.MultiHeadAttention(num_heads=4, key_dim=32, name="fusion_attn")(tok, tok)
        tok = layers.LayerNormalization(name="attn_ln")(layers.Add(name="attn_res")([tok, att]))
        fuse = layers.Flatten(name="fuse")(tok)

    elif fusion == "gating_network":
        p_r, p_w, p_f = proj(rgb_feat, "proj_rgb"), proj(wx_f, "proj_wx"), proj(fl_f, "proj_flow")
        gi = layers.Concatenate(name="gate_in")([rgb_feat, wx_f, fl_f])
        gh = layers.Dense(64, activation="relu", name="gate_h1")(gi)
        gh = layers.Dropout(0.1, name="gate_drop")(gh)
        g3 = layers.Dense(3, activation="softmax", name="branch_gate")(gh)   # weights sum to 1
        tok = layers.Concatenate(axis=1, name="tokens")(
            [as_token(p_r, "tok_rgb"), as_token(p_w, "tok_wx"), as_token(p_f, "tok_flow")])
        weighted = layers.Multiply(name="weighted")([tok, layers.Reshape((3, 1), name="g3_col")(g3)])
        fuse = layers.GlobalAveragePooling1D(name="fuse")(weighted)

    else:
        raise ValueError(f"unknown fusion '{fusion}'")

    # ── Prediction head: identical for every method ──
    z = layers.Dense(128, activation="relu", name="h_d1")(fuse)
    z = layers.Dropout(0.2, name="h_drop1")(z)
    z = layers.Dense(64, activation="relu", name="h_d2")(z)
    z = layers.Dropout(0.1, name="h_drop2")(z)
    out = layers.Dense(1, activation="linear", name="power_delta_norm_out")(z)

    model = Model(inputs=[rgb_in, wx_in, flow_in], outputs=out, name=f"SkyFusion_{fusion}")
    steps = max(1, n_train // C.batch)
    sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        C.lr_base, steps * 20, t_mul=1.5, m_mul=0.85, alpha=1e-6)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(sched, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=C.huber_delta),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                 tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


# ============================================================
# One experiment = one fusion method + one seed
# ============================================================
def scores_from_csv(path):
    d = pd.read_csv(path)
    rmse, mae, r2, skill = mets(d["y_true"], d["y_pred_bc"], d["persist"])
    rm = d["is_ramp"].astype(bool).values
    r_rmse, _, _, r_skill = mets(d["y_true"][rm], d["y_pred_bc"][rm], d["persist"][rm])
    return dict(rmse=rmse, mae=mae, r2=r2, skill=skill, ramp_rmse=r_rmse, ramp_skill=r_skill)


def run_experiment(fusion, seed):
    out_csv = os.path.join(C.save_dir, f"skyfusion_fusion_{fusion}_predictions_seed{seed}.csv")
    if os.path.exists(out_csv):
        print(f"\n[Skip] {fusion} seed {seed}: {out_csv} already exists (delete it to rerun)")
        return dict(fusion=fusion, seed=seed, params=np.nan, **scores_from_csv(out_csv))

    print("\n" + "#" * 70)
    print(f"# FUSION = {fusion}    SEED = {seed}")
    print("#" * 70)
    ds_tr = make_dataset(ix_tr, X_wx_tr, X_flow_tr, y_tr, weights=w, shuffle=True, seed=seed)
    model = build_model(fusion, len(WX_COLS), len(ix_tr), seed)
    print(f"[Model] {fusion}: {model.count_params():,} parameters")

    t0 = time.time()
    history = model.fit(
        ds_tr, validation_data=ds_va, epochs=C.epochs, verbose=2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=C.patience,
                                 restore_best_weights=True, verbose=1)])
    n_ep = len(history.history["loss"])
    print(f"  Done in {time.time()-t0:.0f}s ({n_ep} epochs). "
          f"Best val RMSE = {min(history.history['val_rmse'])*C.plant_mw:.4f} MW")
    if SAVE_MODELS:
        model.save(os.path.join(C.save_dir, f"skyfusion_fusion_{fusion}_seed{seed}.keras"))

    # Predictions: absolute power = persistence + predicted change
    preds_va = model.predict(ds_va, verbose=0).ravel()
    preds_te = model.predict(ds_te, verbose=0).ravel()
    y_va = ix_va["y_power"].astype("float32").values
    y_te = ix_te["y_power"].astype("float32").values
    p_va = ix_va["power_now"].astype("float32").values
    p_te = ix_te["power_now"].astype("float32").values
    raw_va = (p_va + preds_va * C.plant_mw).clip(0, C.plant_mw)
    raw_te = (p_te + preds_te * C.plant_mw).clip(0, C.plant_mw)

    # Hourly bias correction fit on validation, applied to test
    vh = pd.to_datetime(ix_va["time"]).dt.hour.values
    bh = {h: float((raw_va[vh == h] - y_va[vh == h]).mean())
          for h in range(24) if (vh == h).sum() > 10}
    th = pd.to_datetime(ix_te["time"]).dt.hour.values
    pred_te = (raw_te - np.array([bh.get(h, 0.) for h in th])).clip(0, C.plant_mw)

    rmse, mae, r2, skill = mets(y_te, pred_te, p_te)
    is_ramp = ix_te["is_ramp"].values
    r_rmse, _, _, r_skill = mets(y_te[is_ramp], pred_te[is_ramp], p_te[is_ramp])
    s_rmse, _, _, s_skill = mets(y_te[~is_ramp], pred_te[~is_ramp], p_te[~is_ramp])
    print("\n" + "=" * 55)
    print(f"[Test — {fusion}, seed {seed}]")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  Skill={skill:.4f}")
    print(f"  Ramp rows  : RMSE={r_rmse:.4f}  Skill={r_skill:.4f}")
    print(f"  Steady rows: RMSE={s_rmse:.4f}  Skill={s_skill:.4f}")
    print("=" * 55)

    kt_te = ix_te["kt"].values
    cond = np.where(kt_te > 0.6, "clear", np.where(kt_te < 0.4, "cloudy", "mixed"))
    te_df = pd.DataFrame({
        "timestamp": ix_te["time"].values, "y_true": y_te, "y_pred_bc": pred_te,
        "y_pred_raw": raw_te, "persist": p_te, "kt": kt_te, "condition": cond,
        "is_ramp": is_ramp})

    # Gate values (only for the two methods that have a gate)
    try:
        if fusion == "ramp_gate":
            gm = Model(inputs=model.input, outputs=model.get_layer("ramp_gate").output)
            te_df["gate"] = gm.predict(ds_te, verbose=0).ravel()
        elif fusion == "gating_network":
            gm = Model(inputs=model.input, outputs=model.get_layer("branch_gate").output)
            g = gm.predict(ds_te, verbose=0)
            te_df["gate_rgb"], te_df["gate_wx"], te_df["gate_flow"] = g[:, 0], g[:, 1], g[:, 2]
    except Exception as e:
        print(f"[Warn] could not extract gate values: {e}")

    te_df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")
    return dict(fusion=fusion, seed=seed, params=model.count_params(), rmse=rmse, mae=mae,
                r2=r2, skill=skill, ramp_rmse=r_rmse, ramp_skill=r_skill)


# ============================================================
# Run everything, then summarise
# ============================================================
results = [run_experiment(fu, sd) for sd in SEEDS for fu in FUSIONS]
res = pd.DataFrame(results)
res.to_csv(os.path.join(C.save_dir, "fusion_comparison_results.csv"), index=False)

print("\n" + "=" * 86)
print(f"[Summary] mean over seeds {SEEDS}   (run-to-run noise is about 0.02 MW RMSE)")
print("=" * 86)
print(f"{'Fusion':<16} {'Params':>10} {'RMSE':>15} {'MAE':>8} {'R2':>7} {'Skill':>8} "
      f"{'RampRMSE':>9} {'RampSkill':>10}")
for fu in FUSIONS:
    g = res[res["fusion"] == fu]
    pr = f"{int(g['params'].iloc[0]):,}" if g["params"].notna().all() else "-"
    print(f"{fu:<16} {pr:>10} {g['rmse'].mean():>8.4f}±{g['rmse'].std(ddof=0):<6.3f} "
          f"{g['mae'].mean():>8.4f} {g['r2'].mean():>7.4f} {g['skill'].mean():>8.4f} "
          f"{g['ramp_rmse'].mean():>9.3f} {g['ramp_skill'].mean():>10.4f}")
print("\nA difference smaller than ~0.02 MW RMSE is NOT a real difference.")
print("Next: run analysis/gate_analysis.py on the ramp_gate prediction files.")
