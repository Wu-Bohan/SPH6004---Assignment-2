import argparse
import os

import numpy as np
import torch
import yaml

from .common import ensure_dir, make_time_grid, save_json
from .train import load_bundle, run_e4_e5, save_outputs, set_seed


def zero_like(x):
    return np.zeros_like(x, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--mode", choices=["static_only", "static_ts"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_metrics_dir", required=True)
    ap.add_argument("--out_models_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_metrics_dir)
    ensure_dir(args.out_models_dir)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    bundle = load_bundle(args.data_dir, "E4")

                                                     
    if args.mode == "static_only":
        bundle["ts_values"] = zero_like(bundle["ts_values"])
        bundle["ts_mask"] = zero_like(bundle["ts_mask"])
        bundle["ts_delta"] = zero_like(bundle["ts_delta"])
        bundle["x_text_fast"] = zero_like(bundle["x_text_fast"])
    elif args.mode == "static_ts":
        bundle["x_text_fast"] = zero_like(bundle["x_text_fast"])

    time_grid = make_time_grid(bundle["duration"][bundle["idx"]["train"]], n_grid=100)
    pred, risk, surv_bins, art = run_e4_e5(bundle, cfg.get("E4", {}), text_mode="fast")

    bt = np.asarray(art["bin_times"], dtype=float)
    surv = np.vstack([np.interp(time_grid, bt, s, left=1.0, right=s[-1]) for s in surv_bins])
    state = art.pop("state")
    torch.save(state, os.path.join(args.out_models_dir, f"{args.exp_name}.pt"))

    met = save_outputs(
        args.exp_name,
        args.out_metrics_dir,
        bundle,
        time_grid,
        pred,
        risk,
        np.clip(surv, 1e-6, 1.0),
        args.n_bootstrap,
    )
    met["artifact"] = art
    met["artifact"]["ablation_mode"] = args.mode
    save_json(met, os.path.join(args.out_metrics_dir, f"{args.exp_name}.json"))

    print(f"{args.exp_name} completed ({args.mode})")


if __name__ == "__main__":
    main()
