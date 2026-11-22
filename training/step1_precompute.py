# ============================================================
# step1_precompute.py
#
# Precomputes optical flow between consecutive 5-minute sky-camera
# images, so it doesn't get recomputed on every training run.
#
#   -> flows_precomputed_128_1056.npz
#
# Pure OpenCV/NumPy — no TensorFlow, no GPU needed for this step.
# Run once, then run step2_train.py.
#
# (Removed: EfficientNetB4 feature extraction, which used to be
# Part 1 here. Confirmed unused — step2_train.py never loads that
# output. If you bring back an architecture that needs pretrained
# image features, re-add it; for now it was ~20-40 min of compute
# producing a file nothing read.)
# ============================================================
import os, re, glob, time
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import cv2

# ── Paths ────────────────────────────────────────────────────
# NOTE: these are the original developer's paths. Replace both with your own exact paths.
IMAGES_ROOT = "/path/to/sky_images"  # <-- replace with your own sky-image folder path
FLOW_SAVE   = "/path/to/flows_precomputed_128_1056.npz"  # <-- replace with your own output path for the flow file
FLOW_SIZE   = 128   # downsampled further to 48 at training time


# ── Helpers ──────────────────────────────────────────────────
def parse_time_from_name(fname: str) -> Optional[datetime]:
    """Extracts a 14-digit YYYYMMDDHHMMSS timestamp from the start
    of a filename, e.g. '20240516154100_11 LT.jpg'."""
    m = re.match(r"(\d{14})", os.path.basename(fname))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None


def floor5(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def circular_crop(img_bgr: np.ndarray, out_size: int) -> np.ndarray:
    """Crops the circular fisheye region (centred, radius = 46% of the
    smaller dimension) and resizes to out_size x out_size."""
    h, w = img_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.46)
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)
    out = img_bgr[y1:y2, x1:x2]
    return cv2.resize(out, (out_size, out_size), interpolation=cv2.INTER_AREA)


def build_image_map(root: str) -> Dict[datetime, str]:
    """Maps each 5-minute timestamp to its image file path."""
    paths = []
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        paths += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    mp = {}
    for p in sorted(set(paths)):
        dt = parse_time_from_name(p)
        if dt is not None:
            mp[floor5(dt)] = p
    return mp


# ============================================================
# Pre-compute optical flow
# ============================================================
print("=" * 60)
print("Pre-computing optical flow")
print("=" * 60)

if os.path.exists(FLOW_SAVE):
    print(f"[Skip] {FLOW_SAVE} already exists. Delete to re-compute.")
else:
    img_map = build_image_map(IMAGES_ROOT)
    sorted_times = sorted(img_map.keys())
    print(f"[Info] Image map: {len(img_map)} timestamps")

    flow_keys, flow_values = [], []
    skipped = 0
    t0 = time.time()

    for i in range(len(sorted_times) - 1):
        t1, t2 = sorted_times[i], sorted_times[i + 1]

        # Only compute flow for genuinely consecutive 5-min pairs —
        # skip if there's a gap (missing image in between).
        if (t2 - t1) != timedelta(minutes=5):
            skipped += 1
            continue

        p1, p2 = img_map[t1], img_map[t2]
        key = f"{os.path.basename(p1)}__{os.path.basename(p2)}"

        try:
            i1 = circular_crop(cv2.imread(p1), FLOW_SIZE)
            i2 = circular_crop(cv2.imread(p2), FLOW_SIZE)
            if i1 is None or i2 is None:
                skipped += 1
                continue

            g1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            flow_keys.append(key)
            flow_values.append(flow.astype(np.float16))  # float16 halves disk use

        except Exception as e:
            print(f"  [Warn] Flow failed {key}: {e}")
            skipped += 1
            continue

        if i % 2000 == 0:
            print(f"  [{i}/{len(sorted_times)}]  flows={len(flow_keys)}  "
                  f"skipped={skipped}  elapsed={time.time()-t0:.0f}s")

    flow_values = np.stack(flow_values, axis=0)
    print(f"\n[Info] Flow matrix shape: {flow_values.shape}")
    print(f"[Info] Skipped {skipped} non-consecutive/failed pairs")
    np.savez_compressed(FLOW_SAVE, keys=np.array(flow_keys), values=flow_values)
    print(f"[Saved] {FLOW_SAVE} ({os.path.getsize(FLOW_SAVE)/1e6:.1f} MB, "
          f"{time.time()-t0:.0f}s)")

print("\n" + "=" * 60)
print("PRECOMPUTATION COMPLETE")
print(f"  Flows : {FLOW_SAVE}")
print("Now run: step2_train.py")
print("=" * 60)
