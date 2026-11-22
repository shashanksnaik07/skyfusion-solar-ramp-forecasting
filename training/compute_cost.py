"""
Measure the computational cost of a trained model for Table 9 of the paper.

Called once from each training script, right after training finishes. It writes
compute_<name>_seed<seed>.csv with: device, GPU name, parameters, epochs, training
time, time per epoch, inference time per test sample, peak RAM, peak GPU memory.

Inference time: time to predict the whole test set (after one warm-up pass)
divided by the number of test samples.
Peak RAM: peak resident memory of the whole process (data loading + training + inference).
Peak GPU memory: peak memory allocated by TensorFlow on GPU 0 during inference (the counter is reset just before the timed predictions).
Run one model per process, so these peaks belong to that model alone.
"""
import os
import time
import socket
import resource

import numpy as np
import pandas as pd
import tensorflow as tf


def reset_gpu_peak():
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def measure(model, ds_te, n_test, name, train_time_s, epochs, out_dir, seed, has_training=True):
    gpus = tf.config.list_physical_devices("GPU")
    device = "GPU" if gpus else "CPU"
    gpu_name = ""
    if gpus:
        try:
            gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "")
        except Exception:
            pass

    # GPU peak so far = whole training run (counter was reset just before fit)
    gpu_train_gb = np.nan
    if gpus and has_training:
        try:
            gpu_train_gb = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1024 ** 3
        except Exception:
            pass
    reset_gpu_peak()          # from here the GPU peak covers inference only (weights + prediction)
    model.predict(ds_te, verbose=0)                       # warm-up (graph tracing)
    t0 = time.time()
    model.predict(ds_te, verbose=0)
    predict_s = time.time() - t0

    params_total = int(model.count_params())
    params_trained = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024   # kB -> GB
    gpu_infer_gb = np.nan
    if gpus:
        try:
            gpu_infer_gb = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1024 ** 3
        except Exception:
            pass

    row = dict(
        name=name, seed=seed, host=socket.gethostname(), device=device, gpu_name=gpu_name,
        params_total_M=params_total / 1e6, params_trained_M=params_trained / 1e6,
        epochs=int(epochs), train_time_min=train_time_s / 60,
        train_s_per_epoch=train_time_s / max(1, int(epochs)),
        inference_ms_per_sample=predict_s / n_test * 1000,
        ram_peak_GB=ram_gb, gpu_peak_GB=(gpu_train_gb if has_training else gpu_infer_gb),
        gpu_peak_infer_only_GB=gpu_infer_gb, peak_basis=("training+inference" if has_training else "inference only"),
        n_test=int(n_test),
    )
    path = os.path.join(out_dir, f"compute_{name}_seed{seed}.csv")
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"\n[Compute] {name}: {device} {gpu_name} | params {params_total/1e6:.3f} M "
          f"({params_trained/1e6:.3f} M trained) | {epochs} epochs, {train_time_s/60:.1f} min, "
          f"{train_time_s/max(1,int(epochs)):.2f} s/epoch | inference {row['inference_ms_per_sample']:.3f} ms/sample | "
          f"RAM {ram_gb:.2f} GB | GPU {row['gpu_peak_GB']:.2f} GB ({row['peak_basis']}; inference-only GPU {gpu_infer_gb:.2f} GB)")
    print(f"[Compute] saved {path}")
    return row
