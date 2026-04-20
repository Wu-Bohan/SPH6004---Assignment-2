import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter

try:
    from .common import ensure_dir, load_json
except ImportError:
    from common import ensure_dir, load_json

try:
    import shap
    import xgboost as xgb
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


def _load_metrics(metrics_dir: str, exps: List[str]) -> List[Dict]:
    out = []
    for e in exps:
        p = os.path.join(metrics_dir, f"{e}.json")
        if os.path.exists(p):
            out.append(load_json(p))
    return out


def _summary_table(metrics: List[Dict]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        rows.append(
            {
                "exp": m["exp"],
                "train_c_index": m.get("train_c_index"),
                "val_c_index": m.get("val_c_index"),
                "test_c_index": m.get("test_c_index"),
                "test_ibs": m.get("test_ibs"),
                "test_mae_uncensored": m.get("test_mae_uncensored"),
                "test_rmse_uncensored": m.get("test_rmse_uncensored"),
                "test_cindex_ci_low": m.get("test_bootstrap_ci", {}).get("c_index", {}).get("ci95_low"),
                "test_cindex_ci_high": m.get("test_bootstrap_ci", {}).get("c_index", {}).get("ci95_high"),
            }
        )
    df = pd.DataFrame(rows).sort_values("exp")
    return df


def _plot_metric_bars(df: pd.DataFrame, out_fig_dir: str):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("test_c_index", "Test C-index"),
        ("test_ibs", "Test IBS (lower better)"),
        ("test_mae_uncensored", "Test MAE (Uncensored)"),
        ("test_rmse_uncensored", "Test RMSE (Uncensored)"),
    ]
    for ax, (col, title) in zip(axes.flat, metrics):
        sns.barplot(data=df, x="exp", y=col, ax=ax, palette="viridis")
        ax.set_title(title)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, "model_metric_bars.png"), dpi=200)
    plt.close()


def _plot_ablation(df: pd.DataFrame, out_fig_dir: str):
    mapping = {"Static": "E1", "Static+TS": "E3", "Static+TS+TextFast": "E4", "Static+TS+TextStrong": "E5"}
    rows = []
    for stage, exp in mapping.items():
        hit = df[df["exp"] == exp]
        if len(hit):
            rows.append({"stage": stage, "test_c_index": float(hit.iloc[0]["test_c_index"])})
    if not rows:
        return
    adf = pd.DataFrame(rows)
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=adf, x="stage", y="test_c_index", marker="o")
    plt.title("Modality Ablation (C-index)")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, "ablation_cindex.png"), dpi=200)
    plt.close()


def _plot_calibration(metrics_dir: str, best_exp: str, out_fig_dir: str):
    pred = pd.read_parquet(os.path.join(metrics_dir, f"{best_exp}_predictions.parquet"))
    test = pred[pred["split"] == "test"].copy()
    if len(test) < 50:
        return
    test["bin"] = pd.qcut(test["pred_time"], q=10, labels=False, duplicates="drop")
    grp = test.groupby("bin").apply(
        lambda d: pd.Series(
            {
                "pred_mean": d["pred_time"].mean(),
                "obs_mean_uncensored": d[d["event"] == 1]["duration"].mean(),
            }
        )
    ).dropna()
    if len(grp) == 0:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(grp["pred_mean"], grp["obs_mean_uncensored"], c="tab:blue")
    lim_min = min(grp["pred_mean"].min(), grp["obs_mean_uncensored"].min())
    lim_max = max(grp["pred_mean"].max(), grp["obs_mean_uncensored"].max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1)
    plt.xlabel("Predicted Mean Time (h)")
    plt.ylabel("Observed Mean Time (Uncensored, h)")
    plt.title(f"Calibration Plot ({best_exp}, Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, "calibration_plot.png"), dpi=200)
    plt.close()


def _plot_km_risk_groups(metrics_dir: str, best_exp: str, out_fig_dir: str):
    pred = pd.read_parquet(os.path.join(metrics_dir, f"{best_exp}_predictions.parquet"))
    test = pred[pred["split"] == "test"].copy()
    if len(test) < 50:
        return
    q1, q2 = test["risk_score"].quantile([0.33, 0.67]).tolist()
    test["risk_group"] = np.where(test["risk_score"] <= q1, "Low", np.where(test["risk_score"] <= q2, "Mid", "High"))
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 5))
    for g in ["Low", "Mid", "High"]:
        d = test[test["risk_group"] == g]
        if len(d) == 0:
            continue
        kmf.fit(d["duration"], event_observed=d["event"], label=g)
        kmf.plot_survival_function(ci_show=False)
    plt.title(f"Kaplan-Meier by Predicted Risk ({best_exp}, Test)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, "km_risk_groups.png"), dpi=200)
    plt.close()


def _plot_tree_explain(data_dir: str, models_dir: str, out_fig_dir: str):
    if not HAS_SHAP:
        return
    model_path = os.path.join(models_dir, "E3_xgb.json")
    if not os.path.exists(model_path):
        return
    x_static = pd.read_parquet(os.path.join(data_dir, "X_static.parquet")).sort_values("stay_id")
    x_ts = pd.read_parquet(os.path.join(data_dir, "X_ts_stat.parquet")).sort_values("stay_id")
    feat = x_static.merge(x_ts, on="stay_id", how="inner")
    x = feat.drop(columns=["stay_id"])
    if len(x) > 3000:
        x = x.sample(3000, random_state=42)

    bst = xgb.Booster()
    bst.load_model(model_path)
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(x)
    shap_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(shap_abs)[-20:][::-1]
    top_df = pd.DataFrame(
        {"feature": x.columns[top_idx], "mean_abs_shap": shap_abs[top_idx]}
    )
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_df, y="feature", x="mean_abs_shap", palette="magma")
    plt.title("Top-20 SHAP Features (E3 XGBoost-AFT)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, "tree_shap_top20.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--figures_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    ensure_dir(args.figures_dir)
    exps = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    metrics = _load_metrics(args.metrics_dir, exps)
    if not metrics:
        raise RuntimeError("No metrics files found.")
    summary = _summary_table(metrics)
    summary.to_csv(args.out_csv, index=False)

    _plot_metric_bars(summary, args.figures_dir)
    _plot_ablation(summary, args.figures_dir)

    best_exp = summary.sort_values("test_c_index", ascending=False).iloc[0]["exp"]
    _plot_calibration(args.metrics_dir, best_exp, args.figures_dir)
    _plot_km_risk_groups(args.metrics_dir, best_exp, args.figures_dir)
    _plot_tree_explain(args.data_dir, args.models_dir, args.figures_dir)

    print(f"Evaluation completed. Best exp: {best_exp}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
