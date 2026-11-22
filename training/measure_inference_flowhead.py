"""
Table 9, "This work" row: measure inference cost of the ALREADY TRAINED flow-head model.
No training. Rebuilds the data and model exactly as step2_train_flowhead.py does,
loads the saved weights, then times prediction on the test set.

Usage:  python measure_inference_flowhead.py 42
"""
import os, sys, time

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step2_train_flowhead.py")
src = open(SRC).read()
cut = src.index('print("\\n[Train] Starting training...")')   # everything before this = data + model build
exec(compile(src[:cut], SRC, "exec"))                        # defines C, ds_te, ix_te, model, tf ...

from compute_cost import measure

path = os.path.join(C.save_dir, f"skyfusion_flowhead_seed{C.seed}.keras")
print(f"[Load] {path}")
try:
    model.load_weights(path)
except Exception as e:
    print("[Load] load_weights failed, trying load_model:", e)
    model = tf.keras.models.load_model(path, compile=False, safe_mode=False)

# sanity check: parameter count must match the paper (702,562)
print(f"[Model] {model.count_params():,} parameters")

# training columns are not measured here (set to 0 / 1, ignore them in the CSV)
measure(model, ds_te, len(ix_te), "skyfusion_flowhead_infer", 0.0, 1, C.save_dir, C.seed, has_training=False)
