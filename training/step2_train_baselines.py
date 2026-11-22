# ============================================================
# step2_train_baselines.py
# BASELINE MODELS for the SkyFusion paper (Reviewer 1 comment 13,
# Reviewer 2 comment 4): MLP, LSTM, CNN, CNN-LSTM, ConvLSTM, Transformer.
#
# Trains standard deep-learning models on the SAME dataset as the proposed
# model: same days and split, same test rows, same 33 features, same
# change-in-power target, sample weights, Huber loss, learning-rate
# schedule, early stopping, prediction head and hourly bias correction.
# Only the encoder changes. Image baselines get the same sky images
# but NOT optical flow, the 3D-CNN or the Ramp Gate.
#
# The two "5 measured" models (mlp_raw, lstm_raw) use only the measured
# quantities (power, GHI, air temperature, wind speed, wind direction).
# Comparing them with the 33-feature versions shows how much of the
# improvement comes from the physics-derived features.
#
# Run time: the MLP/LSTM models take a few minutes each; the four image
# models take roughly 30-60 minutes each on a GPU. 8 models are available; MODELS selects which to run
# (currently 4 models x 2 seeds = 8 runs). Finished runs are skipped if you stop and restart.
# Edit MODELS / SEEDS in the "Experiment settings" block below.
#
# Based on step2_train_fusion.py (same data pipeline).
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

def make_dataset(ix, X_wx, X_seq, y, spec, weights=None, shuffle=False, seed=42):
    """Builds the tf.data pipeline for one baseline.
    spec["images"] = how many of the most recent frames the model uses (0 = none)."""
    inputs = {"weather": X_wx.astype(np.float32)}
    if spec["seq"]:
        inputs["wxseq"] = X_seq.astype(np.float32)
    k = spec["images"]
    if k:
        inputs["_paths"] = np.array(
            ["|".join(str(ix.iloc[i][f"p{j}"]) for j in range(C.seq_len - k + 1, C.seq_len + 1))
             for i in range(len(ix))])
    tens = (inputs, y.astype(np.float32))
    if weights is not None:
        tens = tens + (weights.astype(np.float32),)
    ds = tf.data.Dataset.from_tensor_slices(tens)
    if shuffle:
        ds = ds.shuffle(2000, seed=seed)
    if k:
        def add_rgb(inp, *rest):
            inp = dict(inp)
            p = inp.pop("_paths")
            rgb = tf.py_function(func=load_frames, inp=[p], Tout=tf.float32)
            rgb.set_shape((k, C.img_size, C.img_size, 3))
            inp["rgb"] = rgb
            return (inp,) + tuple(rest)
        ds = ds.map(add_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(C.batch).prefetch(tf.data.AUTOTUNE)


# ============================================================
# Experiment settings  (EDIT THESE)
# ============================================================
# Baselines (all use the same data, split, target, weights, loss, schedule
# and prediction head as the proposed model; only the encoder differs):
#   mlp          33 features at time t                        -> MLP
#   mlp_raw      5 measured features at time t                -> MLP
#   lstm         33 features over the last 30 min (7 steps)   -> LSTM
#   lstm_raw     5 measured features over the last 30 min     -> LSTM
#   cnn          latest sky image + 33 features               -> 2D CNN + MLP
#   cnn_lstm     7 sky images + 33 features                   -> CNN per frame, LSTM over frames, + MLP
#   convlstm     7 sky images + 33 features                   -> ConvLSTM, + MLP
#   transformer  7 sky images + 33 features                   -> CNN per frame, Transformer over frames, + MLP
# Image baselines do NOT get optical flow, the Ramp Gate or the 3D-CNN.
MODELS = ["mlp", "lstm", "cnn", "cnn_lstm"]
# All available (add any of these back to MODELS to run them too):
#   "mlp_raw", "lstm_raw", "convlstm", "transformer"
SEEDS = [42, 43]
# Run one model per process (clean memory peaks for Table 9):
#   python step2_train_baselines.py mlp 42
if len(sys.argv) > 1: MODELS = [sys.argv[1]]
if len(sys.argv) > 2: SEEDS = [int(sys.argv[2])]
RAW_COLS = ["Power", "GHI", "AirTemp", "Wind speed", "WindDir"]    # the 5 measured quantities
RAW_IDX = [WX_COLS.index(c) for c in RAW_COLS]
T = C.seq_len
SPEC = {
    "mlp":         dict(images=0, seq=False, feats="all"),
    "mlp_raw":     dict(images=0, seq=False, feats="raw"),
    "lstm":        dict(images=0, seq=True,  feats="all"),
    "lstm_raw":    dict(images=0, seq=True,  feats="raw"),
    "cnn":         dict(images=1, seq=False, feats="all"),
    "cnn_lstm":    dict(images=T, seq=False, feats="all"),
    "convlstm":    dict(images=T, seq=False, feats="all"),
    "transformer": dict(images=T, seq=False, feats="all"),
}
NICE = {"mlp": "MLP (33 features)", "mlp_raw": "MLP (5 measured)", "lstm": "LSTM (33 features)",
        "lstm_raw": "LSTM (5 measured)", "cnn": "CNN + MLP", "cnn_lstm": "CNN-LSTM + MLP",
        "convlstm": "ConvLSTM + MLP", "transformer": "Transformer + MLP"}


# ============================================================
# 30-minute feature sequences (for the LSTM baselines)
# ============================================================
print("[Arrays] Building 30-minute feature sequences for the LSTM baselines...")
with tf.device("/CPU:0"):
    F_ALL = np.nan_to_num(norm_wx(df[WX_COLS].values.astype("float32")).numpy(),
                          nan=0., posinf=0., neginf=0.)       # same train-fit scaling as the main model
_pos = pd.Series(np.arange(len(df)), index=pd.DatetimeIndex(df["timestamp"].values))
POS = _pos[~_pos.index.duplicated(keep="last")]

def build_seq_array(ix):
    """(N, 7, n_features): features at t-30, t-25, ..., t. If a time step is missing
    from the table, the nearest earlier (else later) available step is used."""
    t = pd.to_datetime(ix["time"])
    P = np.full((len(ix), T), np.nan)
    for j in range(T):
        back = (T - 1 - j) * 5
        P[:, j] = POS.reindex(pd.DatetimeIndex(t - pd.Timedelta(minutes=back))).values
    for j in range(1, T):                      # fill gaps from earlier steps
        P[:, j] = np.where(np.isnan(P[:, j]), P[:, j - 1], P[:, j])
    for j in range(T - 2, -1, -1):             # leading gaps from later steps
        P[:, j] = np.where(np.isnan(P[:, j]), P[:, j + 1], P[:, j])
    return F_ALL[P.astype(int)]

X_seq_tr, X_seq_va, X_seq_te = build_seq_array(ix_tr), build_seq_array(ix_va), build_seq_array(ix_te)
print(f"[Arrays] sequences: train {X_seq_tr.shape}, val {X_seq_va.shape}, test {X_seq_te.shape}")

y_tr = ix_tr["y_norm"].astype(np.float32).values
y_va = ix_va["y_norm"].astype(np.float32).values
y_te = ix_te["y_norm"].astype(np.float32).values


def pick(X, feats):
    return X if feats == "all" else X[..., RAW_IDX]


# ============================================================
# Building blocks
# ============================================================
def weather_mlp(x):
    """Same MLP as the proposed model's weather branch."""
    w1 = layers.Dense(128, activation="relu")(x)
    w1 = layers.Dropout(0.1)(w1)
    w2 = layers.Dense(128, activation="relu")(w1)
    return layers.Dense(64, activation="relu")(layers.Add()([w1, w2]))

def head(z):
    """Same prediction head as the proposed model."""
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.1)(z)
    return layers.Dense(1, activation="linear")(z)

def frame_encoder(S):
    """2D CNN for one sky image -> 128 values."""
    inp = layers.Input(shape=(S, S, 3))
    x = inp
    for f in (16, 32, 64):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return Model(inp, layers.GlobalAveragePooling2D()(x))

class PosEmb(layers.Layer):
    """Learned position embedding added to a (batch, steps, dim) sequence."""
    def __init__(self, length, dim, **kw):
        super().__init__(**kw)
        self.length, self.dim = length, dim
    def build(self, input_shape):
        self.pos = self.add_weight(name="pos", shape=(self.length, self.dim),
                                   initializer="random_normal", trainable=True)
    def call(self, x):
        return x + self.pos
    def get_config(self):
        return {**super().get_config(), "length": self.length, "dim": self.dim}


def build_baseline(name, wx_dim, n_train, seed):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed); random.seed(seed); np.random.seed(seed)
    S = C.img_size
    spec = SPEC[name]
    wx_in = layers.Input(shape=(wx_dim,), name="weather")

    if name in ("mlp", "mlp_raw"):
        model_in = [wx_in]
        z = weather_mlp(wx_in)

    elif name in ("lstm", "lstm_raw"):
        seq_in = layers.Input(shape=(T, wx_dim), name="wxseq")
        model_in = [seq_in]
        z = layers.LSTM(128)(seq_in)
        z = layers.Dropout(0.2)(z)

    else:                                           # image baselines
        k = spec["images"]
        rgb_in = layers.Input(shape=(k, S, S, 3), name="rgb")
        model_in = [rgb_in, wx_in]

        if name == "cnn":
            x = layers.Reshape((S, S, 3))(rgb_in)               # the single latest frame
            x = frame_encoder(S)(x)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(0.2)(x)

        elif name == "cnn_lstm":
            x = layers.TimeDistributed(frame_encoder(S))(rgb_in)    # (7, 128)
            x = layers.LSTM(128)(x)
            x = layers.Dropout(0.2)(x)

        elif name == "convlstm":
            x = rgb_in
            for f in (16, 32):                                      # shrink 96 -> 24 first
                x = layers.TimeDistributed(layers.Conv2D(f, 3, padding="same", activation="relu"))(x)
                x = layers.TimeDistributed(layers.BatchNormalization())(x)
                x = layers.TimeDistributed(layers.MaxPool2D())(x)
            x = layers.ConvLSTM2D(32, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(0.2)(x)

        elif name == "transformer":
            x = layers.TimeDistributed(frame_encoder(S))(rgb_in)    # (7, 128)
            x = PosEmb(k, 128)(x)
            for _ in range(2):                                      # 2 encoder blocks
                att = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
                x = layers.LayerNormalization()(layers.Add()([x, att]))
                ff = layers.Dense(256, activation="relu")(x)
                ff = layers.Dense(128)(ff)
                x = layers.LayerNormalization()(layers.Add()([x, ff]))
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dropout(0.2)(x)
        else:
            raise ValueError(name)

        z = layers.Concatenate()([x, weather_mlp(wx_in)])

    model = Model(inputs=model_in, outputs=head(z), name=f"baseline_{name}")
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
# One experiment = one baseline + one seed
# ============================================================
def scores_from_csv(path):
    d = pd.read_csv(path)
    rmse, mae, r2, skill = mets(d["y_true"], d["y_pred_bc"], d["persist"])
    rm = d["is_ramp"].astype(bool).values
    r_rmse, _, _, r_skill = mets(d["y_true"][rm], d["y_pred_bc"][rm], d["persist"][rm])
    return dict(rmse=rmse, mae=mae, r2=r2, skill=skill, ramp_rmse=r_rmse, ramp_skill=r_skill)


def run_experiment(name, seed):
    tag = "_table9" if os.environ.get("TABLE9") == "1" else ""   # rerun for timing without touching earlier results
    out_csv = os.path.join(C.save_dir, f"skyfusion_baseline_{name}_predictions_seed{seed}{tag}.csv")
    if os.path.exists(out_csv):
        print(f"\n[Skip] {name} seed {seed}: {out_csv} already exists (delete it to rerun)")
        return dict(model=name, seed=seed, params=np.nan, **scores_from_csv(out_csv))

    print("\n" + "#" * 70)
    print(f"# BASELINE = {name}    SEED = {seed}")
    print("#" * 70)
    spec = SPEC[name]
    Xw = [pick(a, spec["feats"]) for a in (X_wx_tr, X_wx_va, X_wx_te)]
    Xs = [pick(a, spec["feats"]) for a in (X_seq_tr, X_seq_va, X_seq_te)]
    ds_tr = make_dataset(ix_tr, Xw[0], Xs[0], y_tr, spec, weights=w, shuffle=True, seed=seed)
    ds_va = make_dataset(ix_va, Xw[1], Xs[1], y_va, spec)
    ds_te = make_dataset(ix_te, Xw[2], Xs[2], y_te, spec)

    model = build_baseline(name, Xw[0].shape[1], len(ix_tr), seed)
    print(f"[Model] {name}: {model.count_params():,} parameters")

    from compute_cost import reset_gpu_peak, measure
    reset_gpu_peak()
    t0 = time.time()
    history = model.fit(
        ds_tr, validation_data=ds_va, epochs=C.epochs, verbose=2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=C.patience,
                                 restore_best_weights=True, verbose=1)])
    train_s = time.time() - t0
    print(f"  Done in {time.time()-t0:.0f}s ({len(history.history['loss'])} epochs). "
          f"Best val RMSE = {min(history.history['val_rmse'])*C.plant_mw:.4f} MW")

    # Absolute power = persistence + predicted change; hourly bias fit on validation
    raw_va = (ix_va["power_now"].astype("float32").values
              + model.predict(ds_va, verbose=0).ravel() * C.plant_mw).clip(0, C.plant_mw)
    raw_te = (ix_te["power_now"].astype("float32").values
              + model.predict(ds_te, verbose=0).ravel() * C.plant_mw).clip(0, C.plant_mw)
    # Table 9: inference time, peak RAM, peak GPU memory
    measure(model, ds_te, len(ix_te), name, train_s,
            len(history.history['loss']), C.save_dir, seed)
    y_va_mw = ix_va["y_power"].astype("float32").values
    y_te_mw = ix_te["y_power"].astype("float32").values
    p_te = ix_te["power_now"].astype("float32").values
    vh = pd.to_datetime(ix_va["time"]).dt.hour.values
    bh = {h: float((raw_va[vh == h] - y_va_mw[vh == h]).mean())
          for h in range(24) if (vh == h).sum() > 10}
    th = pd.to_datetime(ix_te["time"]).dt.hour.values
    pred_te = (raw_te - np.array([bh.get(h, 0.) for h in th])).clip(0, C.plant_mw)

    rmse, mae, r2, skill = mets(y_te_mw, pred_te, p_te)
    is_ramp = ix_te["is_ramp"].values
    r_rmse, _, _, r_skill = mets(y_te_mw[is_ramp], pred_te[is_ramp], p_te[is_ramp])
    s_rmse, _, _, s_skill = mets(y_te_mw[~is_ramp], pred_te[~is_ramp], p_te[~is_ramp])
    print("\n" + "=" * 55)
    print(f"[Test — {name}, seed {seed}]")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  Skill={skill:.4f}")
    print(f"  Ramp rows  : RMSE={r_rmse:.4f}  Skill={r_skill:.4f}")
    print(f"  Steady rows: RMSE={s_rmse:.4f}  Skill={s_skill:.4f}")
    print("=" * 55)

    kt_te = ix_te["kt"].values
    cond = np.where(kt_te > 0.6, "clear", np.where(kt_te < 0.4, "cloudy", "mixed"))
    pd.DataFrame({
        "timestamp": ix_te["time"].values, "y_true": y_te_mw, "y_pred_bc": pred_te,
        "y_pred_raw": raw_te, "persist": p_te, "kt": kt_te, "condition": cond,
        "is_ramp": is_ramp}).to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")
    return dict(model=name, seed=seed, params=model.count_params(), rmse=rmse, mae=mae,
                r2=r2, skill=skill, ramp_rmse=r_rmse, ramp_skill=r_skill)


# ============================================================
# Run everything, then summarise
# ============================================================
results = [run_experiment(m, sd) for sd in SEEDS for m in MODELS]

# Add the proposed model (Ramp Gate) from step2_train_fusion.py if its files exist
for sd in SEEDS:
    p = os.path.join(C.save_dir, f"skyfusion_fusion_ramp_gate_predictions_seed{sd}.csv")
    if os.path.exists(p):
        d = pd.read_csv(p, parse_dates=["timestamp"])
        if len(d) == len(ix_te):
            results.append(dict(model="ramp_gate", seed=sd, params=np.nan, **scores_from_csv(p)))
        else:
            print(f"[Warn] {p} has {len(d)} rows vs {len(ix_te)} here: different test set, not added")

res = pd.DataFrame(results)
res.to_csv(os.path.join(C.save_dir, "baseline_comparison_results.csv"), index=False)
NICE["ramp_gate"] = "Proposed (Ramp Gate)"
order = [m for m in MODELS if m in set(res["model"])] + (["ramp_gate"] if "ramp_gate" in set(res["model"]) else [])

print("\n" + "=" * 92)
print(f"[Summary] mean over seeds {SEEDS}   (run-to-run noise is about 0.02 MW RMSE)")
print("=" * 92)
print(f"{'Model':<24} {'Params':>10} {'RMSE':>15} {'MAE':>8} {'R2':>7} {'Skill':>8} "
      f"{'RampRMSE':>9} {'RampSkill':>10}")
mean_rmse = {}
for m in order:
    g = res[res["model"] == m]
    mean_rmse[m] = g["rmse"].mean()
    pr = f"{int(g['params'].iloc[0]):,}" if g["params"].notna().all() else "-"
    print(f"{NICE[m]:<24} {pr:>10} {g['rmse'].mean():>8.4f}±{g['rmse'].std(ddof=0):<6.3f} "
          f"{g['mae'].mean():>8.4f} {g['r2'].mean():>7.4f} {g['skill'].mean():>8.4f} "
          f"{g['ramp_rmse'].mean():>9.3f} {g['ramp_skill'].mean():>10.4f}")

def compare(a, b, label):
    if a in mean_rmse and b in mean_rmse:
        d = mean_rmse[b] - mean_rmse[a]
        verdict = "tie (within noise)" if abs(d) < 0.02 else ("better" if d < 0 else "worse")
        print(f"  {label:<58} {mean_rmse[a]:.4f} -> {mean_rmse[b]:.4f}  ({d:+.4f}, {verdict})")

print("\n[What explains the gap? RMSE change in MW; under ~0.02 is noise]")
print("Physics-derived features (5 measured -> 33 features):")
compare("mlp_raw", "mlp", "MLP")
compare("lstm_raw", "lstm", "LSTM")
print("Time history (single time step -> 30-minute sequence):")
compare("mlp", "lstm", "MLP -> LSTM (33 features)")
print("Sky images (features only -> features + images):")
for m in ("cnn", "cnn_lstm", "convlstm", "transformer"):
    compare("mlp", m, f"MLP -> {NICE[m]}")
print("Proposed model vs each image baseline (architecture + optical flow + gate):")
for m in ("cnn", "cnn_lstm", "convlstm", "transformer"):
    compare(m, "ramp_gate", f"{NICE[m]} -> Proposed")
print("\nA difference smaller than ~0.02 MW RMSE is NOT a real difference.")
