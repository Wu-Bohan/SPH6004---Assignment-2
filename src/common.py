import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    return np.asarray(x)


def compute_point_metrics(
    duration: np.ndarray,
    event: np.ndarray,
    pred_time: np.ndarray,
    risk_score: np.ndarray,
) -> Dict[str, float]:
    duration = to_numpy(duration).astype(float)
    event = to_numpy(event).astype(int)
    pred_time = to_numpy(pred_time).astype(float)
    risk_score = to_numpy(risk_score).astype(float)

    finite_pred = np.isfinite(pred_time)
    if np.any(finite_pred):
        fallback_time = float(np.nanmedian(pred_time[finite_pred]))
    else:
        unc = duration[event == 1]
        fallback_time = float(np.nanmedian(unc)) if len(unc) else float(np.nanmedian(duration))
    pred_time = np.where(np.isfinite(pred_time), pred_time, fallback_time)
    pred_time = np.clip(pred_time, 0.1, float(np.nanpercentile(duration, 99.9) * 5.0))

    risk_score = np.where(np.isfinite(risk_score), risk_score, -pred_time)
    risk_score = np.clip(risk_score, -1e6, 1e6)

    cidx = float(concordance_index(duration, -risk_score, event))
    uncensored = event == 1
    if uncensored.sum() > 0:
        mae = float(np.mean(np.abs(duration[uncensored] - pred_time[uncensored])))
        rmse = float(np.sqrt(np.mean((duration[uncensored] - pred_time[uncensored]) ** 2)))
    else:
        mae = float("nan")
        rmse = float("nan")
    return {
        "c_index": cidx,
        "mae_uncensored": mae,
        "rmse_uncensored": rmse,
    }


def kaplan_meier_censor_survival(
    durations: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    times = np.asarray(times, dtype=float)

    censor_event = 1 - events
    order = np.argsort(durations)
    durations_sorted = durations[order]
    censor_sorted = censor_event[order]

    unique_times = np.unique(durations_sorted)
    n = len(durations_sorted)
    at_risk = n
    surv = 1.0
    surv_map = {}
    idx = 0
    for t in unique_times:
        while idx < n and durations_sorted[idx] < t:
            at_risk -= 1
            idx += 1
        d = int(np.sum((durations_sorted == t) & (censor_sorted == 1)))
        if at_risk > 0:
            surv *= (1.0 - d / at_risk)
        surv_map[t] = max(surv, 1e-6)
        at_risk -= int(np.sum(durations_sorted == t))
        idx = int(np.searchsorted(durations_sorted, t, side="right"))

    out = np.ones_like(times, dtype=float)
    sorted_keys = np.array(sorted(surv_map.keys()), dtype=float)
    sorted_vals = np.array([surv_map[k] for k in sorted_keys], dtype=float)
    for i, t in enumerate(times):
        pos = np.searchsorted(sorted_keys, t, side="right") - 1
        if pos >= 0:
            out[i] = sorted_vals[pos]
    return np.clip(out, 1e-6, 1.0)


def integrated_brier_score(
    durations_train: np.ndarray,
    events_train: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    surv_probs_test: np.ndarray,
    times: np.ndarray,
) -> float:
    durations_train = np.asarray(durations_train, dtype=float)
    events_train = np.asarray(events_train, dtype=int)
    durations_test = np.asarray(durations_test, dtype=float)
    events_test = np.asarray(events_test, dtype=int)
    surv_probs_test = np.asarray(surv_probs_test, dtype=float)
    times = np.asarray(times, dtype=float)

    if surv_probs_test.ndim != 2 or surv_probs_test.shape[1] != len(times):
        raise ValueError("surv_probs_test shape must be [n_samples, n_times]")

    g_t = kaplan_meier_censor_survival(durations_train, events_train, times)
    g_y = kaplan_meier_censor_survival(durations_train, events_train, durations_test)

    briers = []
    for k, t in enumerate(times):
        s_hat = surv_probs_test[:, k]
        y_t = (durations_test > t).astype(float)
        w = np.zeros_like(y_t, dtype=float)

        mask1 = durations_test <= t
        if np.any(mask1):
            w[mask1] = events_test[mask1] / np.maximum(g_y[mask1], 1e-6)
        mask2 = durations_test > t
        if np.any(mask2):
            w[mask2] = 1.0 / np.maximum(g_t[k], 1e-6)

        brier_t = np.mean(w * (y_t - s_hat) ** 2)
        briers.append(brier_t)

    briers = np.asarray(briers, dtype=float)
    if len(times) == 1:
        return float(briers[0])
    area = np.trapz(briers, times) / (times[-1] - times[0] + 1e-8)
    return float(area)


def bootstrap_ci(
    duration: np.ndarray,
    event: np.ndarray,
    pred_time: np.ndarray,
    risk_score: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(random_state)
    n = len(duration)
    c_vals = []
    mae_vals = []
    rmse_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        m = compute_point_metrics(
            duration[idx], event[idx], pred_time[idx], risk_score[idx]
        )
        c_vals.append(m["c_index"])
        mae_vals.append(m["mae_uncensored"])
        rmse_vals.append(m["rmse_uncensored"])

    def pack(values: List[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.nanmean(arr)),
            "ci95_low": float(np.nanpercentile(arr, 2.5)),
            "ci95_high": float(np.nanpercentile(arr, 97.5)),
        }

    return {
        "c_index": pack(c_vals),
        "mae_uncensored": pack(mae_vals),
        "rmse_uncensored": pack(rmse_vals),
    }


def expected_time_from_survival(times: np.ndarray, surv_probs: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    surv_probs = np.asarray(surv_probs, dtype=float)
    if surv_probs.ndim != 2:
        raise ValueError("surv_probs must be 2D")
    if len(times) < 2:
        return np.repeat(times[0], surv_probs.shape[0])
    dt = np.diff(times, prepend=times[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    expected = np.sum(surv_probs * dt[None, :], axis=1)
    return expected


def make_time_grid(durations: np.ndarray, n_grid: int = 100) -> np.ndarray:
    durations = np.asarray(durations, dtype=float)
    t_min = max(0.5, float(np.nanpercentile(durations, 1)))
    t_max = float(np.nanpercentile(durations, 99.5))
    if t_max <= t_min:
        t_max = t_min + 1.0
    return np.linspace(t_min, t_max, n_grid)
