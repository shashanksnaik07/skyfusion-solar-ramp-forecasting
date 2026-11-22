"""
Print the LaTeX rows of Table 9 from the compute_*.csv files written by the training scripts.

Usage:  python fill_table9.py compute_*.csv
"""
import sys
import pandas as pd

LABEL = {"skyfusion_flowhead": "This work", "mlp": "MLP", "lstm": "LSTM",
         "mlp_raw": "MLP", "lstm_raw": "LSTM",
         "skyfusion_flowhead_infer": "This work"}

for path in sys.argv[1:]:
    r = pd.read_csv(path).iloc[0]
    gpu = "--" if pd.isna(r["gpu_peak_GB"]) else f"{r['gpu_peak_GB']:.2f}"
    print(f"    {LABEL.get(r['name'], r['name'])} & {r['device']} & {r['params_total_M']:.3f} & "
          f"{r['params_trained_M']:.3f} & {int(r['epochs'])} & {r['train_time_min']:.1f} & "
          f"{r['train_s_per_epoch']:.1f} & {r['inference_ms_per_sample']:.3f} & "
          f"{r['ram_peak_GB']:.2f} & {gpu} \\\\")
    print(f"    % GPU: {r['gpu_name']}  host: {r['host']}  seed: {int(r['seed'])}  peak basis: {r['peak_basis']}")
