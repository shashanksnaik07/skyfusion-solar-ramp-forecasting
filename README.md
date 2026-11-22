# SkyFusion: 10-Minute-Ahead Solar Power Ramp Forecasting

Predicts the **change** in a solar plant's power output 10 minutes ahead from
a 30-minute sequence of sky-camera images, the optical flow (cloud motion)
between them, and 33 weather / irradiance / PV features. The target is the
power change, not the absolute power level, so "no change" is the default
prediction and the model only has to learn deviations from it.

> **Data are not included.** The sky images and the plant/sensor data cannot be
> shared. This repository contains the code only. See [Data layout](#data-layout)
> for the exact format the scripts expect.

---

## Model

Three branches are fused into one prediction:

- **Image branch:** a 3D-CNN over 7 frames (5 minutes apart).
- **Weather branch:** a dense network over the 33 engineered features.
- **Flow branch:** a 2D-CNN over the precomputed optical-flow field. In
  `step2_train_flowhead.py` the flow branch has an extra Dense(128) + Dropout(0.2)
  after global pooling.
- **Ramp Gate:** a learned scalar that weights the image and flow branches per row.

Training uses the Huber loss (delta = 0.1), Adam with a cosine-decay-with-restarts
schedule, early stopping (patience 10) and sample weights that favour ramp and
cloudy rows. Predictions get a small hourly bias correction fitted on the
validation set. Days are split 1-in-5 interleaved into train / validation / test.

---

## Repository layout

```
training/
  step1_precompute.py            optical flow between consecutive sky images (run once)
  step2_train.py                 base SkyFusion model
  step2_train_flowhead.py        base model + Dense(128)/Dropout(0.2) in the flow branch
  step2_train_fusion.py          same model with five fusion methods (ramp gate, concat, add, attention, gating network)
  step2_train_baselines.py       baselines: MLP, LSTM, CNN, CNN-LSTM (and ConvLSTM, Transformer)
  compute_cost.py                measures inference time and peak memory
  measure_inference_flowhead.py  inference cost of a saved flow-head model, no retraining
  fill_table9.py                 prints LaTeX rows from the compute_*.csv files
analysis/
  make_paper_tables.py           metrics tables
  smart_persistence.py           smart-persistence baseline
  ensemble_average.py            averages predictions over seeds
  flow_branch_ablation.py        flow branch trained alone
  gate_analysis.py               Ramp Gate behaviour
  check_resolution_ceiling.py    image resolution / crop check
  check_flagged_days_data.py     checks whether high-error days have a data problem
  compare_runs.py                compares two prediction files
  plot_day_comparison.py         actual vs predicted power for one day
```

---

## Requirements

```
pip install -r requirements.txt
```

TensorFlow, OpenCV, pandas, NumPy, pvlib, scikit-learn, matplotlib. A GPU is
used by the training scripts when available and is not needed for step 1.

---

## Data layout

Every script has a configuration block at the top with `/path/to/...`
placeholders. Edit these to your own paths before running.

**Sky images** (`/path/to/sky_images`): one photo every 5 minutes. File names
start with a 14-digit timestamp `YYYYMMDDHHMMSS`; anything after it is ignored.
Subfolders are searched recursively. Photos are circular fisheye images; a
centred circular crop (46% of the smaller image side) is used.

**Sensor CSV** (`/path/to/POA_Power_cloudiness_aoi_pvwatts_merged_full.csv`):
tab-separated, one row per 5-minute timestamp, with at least:

| Column | Meaning |
|---|---|
| `timestamp` | datetime |
| `GHI` | measured global horizontal irradiance (W/m²) |
| `ghi_cs` | clear-sky GHI |
| `Power` | measured AC power (MW) |
| `Cloudiness` | sky-image cloudiness index |
| `AirTemp`, `cell_temp`, `Wind speed`, `WindDir` | weather |
| `effective_POA` | fallback POA if `pvfactors` is not installed |
| `Pac_modelchain` (or `Pac_cs`) | model-chain reference power |

**Optical flow** (`/path/to/flows_precomputed_128_1056.npz`): written by step 1,
read by step 2.

**Output folder** (`/path/to/output`): models, predictions and `compute_*.csv` files.

---

## Running

```bash
# 1. optical flow (once)
python training/step1_precompute.py

# 2. train and evaluate (seed as the first argument)
python training/step2_train_flowhead.py 42

# baselines, one model and seed per process
python training/step2_train_baselines.py mlp 42
python training/step2_train_baselines.py lstm 42
```

Each training script saves the model and a predictions CSV with RMSE, MAE, R²
and skill against persistence, broken down by sky condition and ramp vs steady rows.

### Computational cost

`step2_train_flowhead.py` and `step2_train_baselines.py` write
`compute_<model>_seed<seed>.csv` (parameters, epochs, training time, inference
time per test sample, peak RAM, peak GPU memory). Inference time is the time to
predict the whole test set, after one warm-up pass, divided by the number of test
samples. To measure an already trained flow-head model without retraining:

```bash
python training/measure_inference_flowhead.py 42
python training/fill_table9.py compute_*.csv
```

Run one model per process, on the same GPU, so the memory peaks belong to that
model alone.

---

## Reproducibility notes

- Seeds are set, but GPU runs are not deterministic: repeated runs with the same
  seed differ by about 0.02 MW in RMSE.
- The split is by day, so no day appears in more than one of train, validation and test.
- The flow-head script overwrites its saved `.keras` file on each run. Back it up
  first if you want to keep several models.
