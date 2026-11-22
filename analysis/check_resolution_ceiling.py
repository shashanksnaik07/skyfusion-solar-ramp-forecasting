# ============================================================
# check_resolution_ceiling.py
#
# Purpose: check whether your 64x64 image size and precomputed
# optical flow are throwing away the detail needed to size a
# ramp accurately — WITHOUT training anything. Pure data check.
#
# Run this on the same machine where your images and flow file
# live. It reads directly from disk, no model involved.
# ============================================================

import os, glob, random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # no display needed, saves straight to file
import matplotlib.pyplot as plt

# ── Match these to your training script's config ───────────────
# NOTE: these are the original developer's paths. Replace both with
# your own exact paths before running.
IMAGES_ROOT = "/path/to/sky_images"          # <-- replace with your own images folder path
FLOW_PATH   = "/path/to/flows_precomputed_128_1056.npz"  # <-- replace with your own flow file path
OUT_DIR     = "/path/to/output/resolution_check"  # <-- replace with your own output folder path
CX, CY, R   = 0.5, 0.5, 0.46   # same crop config as the training script
TRAIN_SIZE  = 64                # what the model actually sees
SEED        = 42

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

# ── 1. Native resolution vs. what the model actually sees ──────
print("[1/4] Checking native image resolution...")
all_paths = sorted(glob.glob(os.path.join(IMAGES_ROOT, "**", "*.jpg"), recursive=True))
if not all_paths:
    raise SystemExit(f"No images found under {IMAGES_ROOT} — check the path.")
print(f"  Found {len(all_paths)} images total")

sample_for_shape = random.sample(all_paths, min(50, len(all_paths)))
shapes = []
for p in sample_for_shape:
    img = cv2.imread(p)
    if img is not None:
        shapes.append(img.shape[:2])
shapes = np.array(shapes)
h_native, w_native = shapes[0]

print(f"  Native resolution (H,W): mean={shapes.mean(axis=0).round(1)}, "
      f"min={shapes.min(axis=0)}, max={shapes.max(axis=0)}")

crop_diam_px = int(min(h_native, w_native) * R * 2)
downsample_factor = crop_diam_px / TRAIN_SIZE
print(f"  Crop covers ~{crop_diam_px}px across (that's {R*2*100:.0f}% of the "
      f"smaller image dimension)")
print(f"  Then resized down to {TRAIN_SIZE}x{TRAIN_SIZE} for training")
print(f"  Downsampling factor: ~{downsample_factor:.1f}x per side "
      f"({downsample_factor**2:.0f}x fewer total pixels than the native crop)")

# ── 2. Visual side-by-side: native crop vs. actual model input ─
print("\n[2/4] Building visual comparison...")
n_examples = 6
idxs = np.linspace(0, len(all_paths) - 1, n_examples).astype(int)
picks = [all_paths[i] for i in idxs]

fig, axes = plt.subplots(n_examples, 3, figsize=(9, 3 * n_examples))
for row, p in enumerate(picks):
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    cx, cy = int(w * CX), int(h * CY)
    r = int(min(h, w) * R)
    crop = img[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
    if crop.size == 0:
        continue

    small = cv2.resize(crop, (TRAIN_SIZE, TRAIN_SIZE), interpolation=cv2.INTER_AREA)
    small_blown_up = cv2.resize(small, (crop.shape[1], crop.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

    for col, (im, title) in enumerate([
        (crop, f"Native crop ({crop.shape[1]}x{crop.shape[0]})"),
        (small, f"Actual model input ({TRAIN_SIZE}x{TRAIN_SIZE})"),
        (small_blown_up, "Model input, blown back up to compare"),
    ]):
        axes[row, col].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(title, fontsize=9)
        axes[row, col].axis("off")

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "resolution_comparison.png")
plt.savefig(out_png, dpi=110)
plt.close()
print(f"  Saved: {out_png}")
print("  Open this and look at cloud edges directly. If they're mush in the")
print("  middle column, that's your answer regardless of any number below.")

# ── 3. Numeric detail-loss score (edge energy retained) ─────────
print("\n[3/4] Measuring how much fine detail survives downsampling...")

def edge_energy(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx**2 + gy**2).mean())

sample_for_edges = random.sample(all_paths, min(100, len(all_paths)))
edge_ratios = []
for p in sample_for_edges:
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    cx, cy = int(w * CX), int(h * CY)
    r = int(min(h, w) * R)
    crop = img[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
    if crop.size == 0:
        continue
    gray_native = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(crop, (TRAIN_SIZE, TRAIN_SIZE), interpolation=cv2.INTER_AREA)
    gray_small_up = cv2.resize(
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
        (gray_native.shape[1], gray_native.shape[0]),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    e_native = edge_energy(gray_native)
    e_small = edge_energy(gray_small_up)
    if e_native > 1e-6:
        edge_ratios.append(e_small / e_native)

edge_ratios = np.array(edge_ratios)
print(f"  Average fine-detail retained after downsampling: "
      f"{edge_ratios.mean()*100:.1f}%")
print(f"  (100% = no detail lost, well under 50% = a lot of cloud-edge")
print(f"  structure is being destroyed before the model ever sees it)")

# ── 4. Is the precomputed optical flow actually detailed? ───────
print("\n[4/4] Checking optical flow field resolution and structure...")
flow_npz = np.load(FLOW_PATH)
flow_store = dict(zip(flow_npz["keys"].tolist(), flow_npz["values"]))
all_keys = list(flow_store.keys())
sample_keys = random.sample(all_keys, min(200, len(all_keys)))

native_flow_shape = flow_store[sample_keys[0]].shape
mags = []
internal_stds = []
for k in sample_keys:
    f = flow_store[k]
    fx, fy = f[..., 0], f[..., 1]
    mag = np.sqrt(fx**2 + fy**2)
    mags.append(mag.mean())
    internal_stds.append(mag.std())

mags = np.array(mags)
internal_stds = np.array(internal_stds)
near_uniform_pct = 100 * np.mean(internal_stds < 0.01)

print(f"  Stored flow field resolution: {native_flow_shape[0]}x{native_flow_shape[1]}")
print(f"  (Training script resizes this down further to 48x48)")
print(f"  Mean flow magnitude across {len(sample_keys)} sampled pairs: {mags.mean():.4f}")
print(f"  % of flow fields with almost no internal spatial variation: "
      f"{near_uniform_pct:.1f}%")
print("  High % here = most flow fields are a blurry 'something moved'")
print("  signal, not detailed cloud-edge motion. That would explain why")
print("  the flow branch never earned its keep despite having 100% coverage.")

print("\n" + "="*55)
print("[Summary] What to look at, in order of how direct the evidence is:")
print("  1. Open resolution_comparison.png and look at it yourself.")
print("  2. Edge-detail-retained %: lower = stronger case resolution is")
print("     the ceiling, not the training setup.")
print("  3. Flow near-uniform %: higher = flow branch structurally can't")
print("     see fine cloud motion, regardless of model tuning.")
print("="*55)
