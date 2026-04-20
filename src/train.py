import argparse
import os
import pickle
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import yaml
from lifelines import CoxPHFitter
from scipy.stats import norm
from torch.utils.data import DataLoader, Dataset

try:
    from .common import (
        bootstrap_ci,
        compute_point_metrics,
        ensure_dir,
        expected_time_from_survival,
        integrated_brier_score,
        load_json,
        make_time_grid,
        save_json,
    )
except ImportError:
    from common import (
        bootstrap_ci,
        compute_point_metrics,
        ensure_dir,
        expected_time_from_survival,
        integrated_brier_score,
        load_json,
        make_time_grid,
        save_json,
    )

warnings.filterwarnings("ignore")

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    HAS_SKSURV = True
except Exception:
    HAS_SKSURV = False


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def step_surv(pred_time: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    return (time_grid[None, :] < pred_time[:, None]).astype(float)


def load_bundle(data_dir: str, exp: str):
    cohort = pd.read_parquet(os.path.join(data_dir, "cohort.parquet")).sort_values("stay_id")
    splits = load_json(os.path.join(data_dir, "splits.json"))
    split_map = {int(sid): k for k, ids in splits.items() for sid in ids}
    cohort["split"] = cohort["stay_id"].astype(int).map(split_map)

    stay = cohort["stay_id"].astype(int).to_numpy()
    duration = cohort["duration"].astype(float).to_numpy()
    event = cohort["event"].astype(int).to_numpy()
    split = cohort["split"].astype(str).to_numpy()

    def align(path):
        x = pd.read_parquet(path).set_index("stay_id").loc[stay]
        return x.to_numpy(dtype=np.float32)

    need_static = exp in {"E1", "E2", "E3", "E4", "E5"}
    need_ts_stat = exp in {"E2", "E3"}
    need_seq = exp in {"E4", "E5"}
    need_text_fast = exp in {"E4", "E5"}
    need_text_strong = exp in {"E5"}

    x_static = align(os.path.join(data_dir, "X_static.parquet")) if need_static else None
    x_ts_stat = align(os.path.join(data_dir, "X_ts_stat.parquet")) if need_ts_stat else None
    x_text_fast = align(os.path.join(data_dir, "X_text_fast.parquet")) if need_text_fast else None

    if need_text_strong:
        strong_path = os.path.join(data_dir, "X_text_strong.parquet")
        x_text_strong = align(strong_path) if os.path.exists(strong_path) else x_text_fast.copy()
    else:
        x_text_strong = None

    if need_seq:
        seq = np.load(os.path.join(data_dir, "X_ts_seq.npz"), allow_pickle=True)
        seq_sid = seq["stay_id"].astype(int)
        order = pd.Series(np.arange(len(seq_sid)), index=seq_sid).loc[stay].to_numpy()
        ts_values = seq["values"][order].astype(np.float32)
        ts_mask = seq["mask"][order].astype(np.float32)
        ts_delta = seq["delta"][order].astype(np.float32)
    else:
        ts_values = None
        ts_mask = None
        ts_delta = None

    idx = {
        "train": np.where(split == "train")[0],
        "val": np.where(split == "val")[0],
        "test": np.where(split == "test")[0],
    }
    return {
        "stay": stay, "duration": duration, "event": event, "split": split, "idx": idx,
        "x_static": x_static, "x_ts_stat": x_ts_stat, "x_text_fast": x_text_fast,
        "x_text_strong": x_text_strong, "ts_values": ts_values, "ts_mask": ts_mask, "ts_delta": ts_delta,
    }


def save_outputs(exp, out_metrics_dir, bundle, time_grid, pred_time, risk, surv, n_bootstrap):
    idx = bundle["idx"]
    duration = bundle["duration"]
    event = bundle["event"]
    metrics = {"exp": exp}
    for sp in ["train", "val", "test"]:
        ii = idx[sp]
        m = compute_point_metrics(duration[ii], event[ii], pred_time[ii], risk[ii])
        metrics[f"{sp}_c_index"] = m["c_index"]
        metrics[f"{sp}_mae_uncensored"] = m["mae_uncensored"]
        metrics[f"{sp}_rmse_uncensored"] = m["rmse_uncensored"]
    metrics["test_ibs"] = integrated_brier_score(
        durations_train=duration[idx["train"]],
        events_train=event[idx["train"]],
        durations_test=duration[idx["test"]],
        events_test=event[idx["test"]],
        surv_probs_test=surv[idx["test"]],
        times=time_grid,
    )
    metrics["test_bootstrap_ci"] = bootstrap_ci(
        duration[idx["test"]],
        event[idx["test"]],
        pred_time[idx["test"]],
        risk[idx["test"]],
        n_bootstrap=n_bootstrap,
    )
    pred_df = pd.DataFrame({
        "stay_id": bundle["stay"],
        "duration": duration,
        "event": event,
        "split": bundle["split"],
        "pred_time": pred_time,
        "risk_score": risk,
    })
    pred_df.to_parquet(os.path.join(out_metrics_dir, f"{exp}_predictions.parquet"), index=False)
    np.savez_compressed(
        os.path.join(out_metrics_dir, f"{exp}_survival.npz"),
        times=time_grid.astype(np.float32),
        surv=surv.astype(np.float32),
        stay_id=bundle["stay"].astype(np.int64),
    )
    save_json(metrics, os.path.join(out_metrics_dir, f"{exp}.json"))
    return metrics


def run_e0(bundle, time_grid):
    idx = bundle["idx"]
    med = float(np.median(bundle["duration"][idx["train"]][bundle["event"][idx["train"]] == 1]))
    pred = np.repeat(med, len(bundle["stay"]))
    risk = -pred
    surv = step_surv(pred, time_grid)
    return pred, risk, surv, {"median_time": med}


def run_e1(bundle, time_grid, cfg):
    x = bundle["x_static"]
    all_cols = [f"f{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=all_cols)
    df["duration"] = bundle["duration"]
    df["event"] = bundle["event"]
    train = df.iloc[bundle["idx"]["train"]]
    feat_var = train[all_cols].var(axis=0)
    keep = feat_var[feat_var > 1e-8].index.tolist()
    max_features = int(cfg.get("max_features", 150))
    if len(keep) > max_features:
        keep = feat_var.sort_values(ascending=False).head(max_features).index.tolist()
    cols = keep
    use_cox = bool(cfg.get("use_cox", False))
    if use_cox:
        cph = CoxPHFitter(
            penalizer=float(cfg.get("penalizer", 1.0)),
            l1_ratio=float(cfg.get("l1_ratio", 0.0)),
        )
        try:
            max_rows = int(cfg.get("max_train_rows", 20000))
            train_fit = train[cols + ["duration", "event"]]
            if len(train_fit) > max_rows:
                train_fit = train_fit.sample(max_rows, random_state=42)
            cph.fit(train_fit, duration_col="duration", event_col="event", show_progress=False)
            risk = cph.predict_partial_hazard(df[cols]).to_numpy().reshape(-1)
            pred = cph.predict_median(df[cols]).to_numpy().reshape(-1)
            pred = np.where(np.isfinite(pred), pred, np.median(bundle["duration"][bundle["idx"]["train"]]))
            surv = cph.predict_survival_function(df[cols], times=time_grid).T.to_numpy()
            return pred, risk, np.clip(surv, 1e-6, 1.0), {"model": "coxph", "coef_size": int(len(cph.params_)), "selected_features": int(len(cols))}
        except Exception as e:
            cox_err = str(e)[:200]
    else:
        cox_err = "cox_disabled_by_default_for_resource_stability"

    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=10.0, random_state=42)
    reg.fit(train[cols], np.log1p(train["duration"]))
    pred = np.expm1(reg.predict(df[cols]))
    risk = -pred
    surv = step_surv(pred, time_grid)
    return pred, risk, surv, {"model": "ridge_aft_fallback", "fallback_reason": cox_err, "selected_features": int(len(cols))}


def run_e2(bundle, time_grid, cfg):
    x = np.concatenate([bundle["x_static"], bundle["x_ts_stat"]], axis=1)
    idx = bundle["idx"]
    x_tr, x_va, x_all = x[idx["train"]], x[idx["val"]], x
    y_tr_d = bundle["duration"][idx["train"]]
    y_tr_e = bundle["event"][idx["train"]]
    y_va_d = bundle["duration"][idx["val"]]
    y_va_e = bundle["event"][idx["val"]]
    use_sksurv = bool(cfg.get("use_sksurv", False))
    if (not HAS_SKSURV) or (not use_sksurv):
        from sklearn.ensemble import RandomForestRegressor
        n_trials = int(cfg.get("n_trials", 30))

        def obj_rf(trial):
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 700),
                "max_depth": trial.suggest_int("max_depth", 4, 24),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_float("max_features", 0.2, 1.0),
                "n_jobs": -1,
                "random_state": 42,
            }
            m = RandomForestRegressor(**p)
            m.fit(x_tr, np.log1p(y_tr_d))
            pred = np.expm1(m.predict(x_va))
            return compute_point_metrics(y_va_d, y_va_e, pred, -pred)["c_index"]

        study = optuna.create_study(direction="maximize")
        study.optimize(obj_rf, n_trials=n_trials, show_progress_bar=False)
        p = study.best_params
        p.update({"n_jobs": -1, "random_state": 42})
        m = RandomForestRegressor(**p)
        m.fit(x_tr, np.log1p(y_tr_d))
        pred = np.expm1(m.predict(x_all))
        risk = -pred
        surv = step_surv(pred, time_grid)
        return pred, risk, surv, {"fallback": "rf_regressor_tuned", "best_params": p, "best_value": float(study.best_value), "__model__": m}

    y_tr = Surv.from_arrays(y_tr_e.astype(bool), y_tr_d.astype(float))
    n_trials = int(cfg.get("n_trials", 30))

    def obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 4, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 4, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
            "n_jobs": -1, "random_state": 42,
        }
        m = RandomSurvivalForest(**p)
        m.fit(x_tr, y_tr)
        risk = m.predict(x_va)
        return compute_point_metrics(y_va_d, y_va_e, np.repeat(np.median(y_tr_d), len(risk)), risk)["c_index"]

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    p = study.best_params
    p.update({"n_jobs": -1, "random_state": 42})
    m = RandomSurvivalForest(**p)
    m.fit(x_tr, y_tr)
    risk = m.predict(x_all)
    surv_raw = m.predict_survival_function(x_all, return_array=True)
    raw_t = m.unique_times_
    surv = np.vstack([np.interp(time_grid, raw_t, s, left=1.0, right=s[-1]) for s in surv_raw])
    pred = expected_time_from_survival(time_grid, surv)
    return pred, risk, np.clip(surv, 1e-6, 1.0), {"best_params": p, "best_value": float(study.best_value), "__model__": m}


def run_e3(bundle, time_grid, cfg):
    x = np.concatenate([bundle["x_static"], bundle["x_ts_stat"]], axis=1)
    idx = bundle["idx"]
    x_tr, x_va, x_all = x[idx["train"]], x[idx["val"]], x
    d_tr = bundle["duration"][idx["train"]]
    e_tr = bundle["event"][idx["train"]]
    d_va = bundle["duration"][idx["val"]]
    e_va = bundle["event"][idx["val"]]

    def make_dm(X, D, E):
        dm = xgb.DMatrix(X)
        dm.set_float_info("label_lower_bound", D)
        up = D.copy()
        up[E == 0] = np.inf
        dm.set_float_info("label_upper_bound", up)
        return dm

    dtrain = make_dm(x_tr, d_tr.astype(float), e_tr.astype(int))
    dval = make_dm(x_va, d_va.astype(float), e_va.astype(int))
    dall = make_dm(x_all, bundle["duration"].astype(float), bundle["event"].astype(int))

    n_trials = int(cfg.get("n_trials", 30))
    rounds = int(cfg.get("num_boost_round", 1000))
    early = int(cfg.get("early_stopping_rounds", 50))

    def obj(trial):
        p = {
            "objective": "survival:aft", "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": trial.suggest_float("scale", 0.5, 2.0),
            "learning_rate": trial.suggest_float("eta", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "lambda": trial.suggest_float("lambda", 1e-4, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "tree_method": "hist", "seed": 42,
        }
        bst = xgb.train(p, dtrain, num_boost_round=rounds, evals=[(dval, "val")], early_stopping_rounds=early, verbose_eval=False)
        mu = bst.predict(dval)
        pred = np.exp(np.clip(mu, -3.0, 9.0))
        return compute_point_metrics(d_va, e_va, pred, -mu)["c_index"]

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    scale = float(bp.pop("scale"))
    p = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": scale,
        "learning_rate": float(bp["eta"]),
        "max_depth": int(bp["max_depth"]),
        "subsample": float(bp["subsample"]),
        "colsample_bytree": float(bp["colsample_bytree"]),
        "min_child_weight": float(bp["min_child_weight"]),
        "lambda": float(bp["lambda"]),
        "alpha": float(bp["alpha"]),
        "tree_method": "hist", "seed": 42,
    }
    bst = xgb.train(p, dtrain, num_boost_round=rounds, evals=[(dval, "val")], early_stopping_rounds=early, verbose_eval=False)
    mu = bst.predict(dall)
    log_t = np.clip(mu + 0.5 * scale * scale, -3.0, 9.0)
    pred = np.exp(log_t)
    risk = -mu
    z = (np.log(np.maximum(time_grid[None, :], 1e-8)) - np.clip(mu[:, None], -5.0, 10.0)) / max(scale, 1e-4)
    surv = np.clip(1.0 - norm.cdf(z), 1e-6, 1.0)
    return pred, risk, surv, {"best_params": p, "best_value": float(study.best_value), "__model__": bst}


class FusionDataset(Dataset):
    def __init__(self, xs, tv, tm, td, xt, bi, ev):
        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.tv = torch.tensor(tv, dtype=torch.float32)
        self.tm = torch.tensor(tm, dtype=torch.float32)
        self.td = torch.tensor(td, dtype=torch.float32)
        self.xt = torch.tensor(xt, dtype=torch.float32)
        self.bi = torch.tensor(bi, dtype=torch.long)
        self.ev = torch.tensor(ev.astype(np.float32), dtype=torch.float32)
    def __len__(self): return len(self.bi)
    def __getitem__(self, i): return self.xs[i], self.tv[i], self.tm[i], self.td[i], self.xt[i], self.bi[i], self.ev[i]


class FusionNet(nn.Module):
    def __init__(self, ds, dv, dt, hidden, bins, dropout):
        super().__init__()
        self.s = nn.Sequential(nn.Linear(ds, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.g = nn.GRU(input_size=dv * 3, hidden_size=hidden, batch_first=True)
        self.x = nn.Sequential(nn.Linear(dt, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.w = nn.Linear(hidden * 3, 3)
        self.h = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, bins))
    def forward(self, xs, tv, tm, td, xt):
        zs = self.s(xs)
        _, hg = self.g(torch.cat([tv, tm, td], dim=-1))
        zt = hg[-1]
        zx = self.x(xt)
        a = torch.softmax(self.w(torch.cat([zs, zt, zx], dim=-1)), dim=-1)
        z = a[:, 0:1] * zs + a[:, 1:2] * zt + a[:, 2:3] * zx
        return self.h(z), a


def nll(logits, bi, ev, eps=1e-7):
    h = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    l1 = torch.log(1.0 - h)
    cs = torch.cumsum(l1, dim=1)
    j = bi.view(-1, 1)
    lsp = torch.where(j > 0, cs.gather(1, (j - 1).clamp(min=0)), torch.zeros_like(j, dtype=logits.dtype))
    lhj = torch.log(h.gather(1, j))
    le = -(lsp + lhj).squeeze(1)
    lc = -cs.gather(1, j).squeeze(1)
    return (ev * le + (1.0 - ev) * lc).mean()


@torch.no_grad()
def infer(model, xs, tv, tm, td, xt, bs, device):
    model.eval()
    hz, gt = [], []
    for i in range(0, len(xs), bs):
        s = torch.tensor(xs[i:i + bs], dtype=torch.float32, device=device)
        v = torch.tensor(tv[i:i + bs], dtype=torch.float32, device=device)
        m = torch.tensor(tm[i:i + bs], dtype=torch.float32, device=device)
        d = torch.tensor(td[i:i + bs], dtype=torch.float32, device=device)
        x = torch.tensor(xt[i:i + bs], dtype=torch.float32, device=device)
        logit, a = model(s, v, m, d, x)
        hz.append(torch.sigmoid(logit).cpu().numpy())
        gt.append(a.cpu().numpy())
    return np.vstack(hz), np.vstack(gt)


def run_e4_e5(bundle, cfg, text_mode):
    idx = bundle["idx"]
    xs = bundle["x_static"]
    xt = bundle["x_text_fast"] if text_mode == "fast" else bundle["x_text_strong"]
    tv, tm, td = bundle["ts_values"], bundle["ts_mask"], bundle["ts_delta"]
    ytr, etr = bundle["duration"][idx["train"]], bundle["event"][idx["train"]]
    yva, eva = bundle["duration"][idx["val"]], bundle["event"][idx["val"]]

    bins = int(cfg.get("n_bins", 64))
    max_t = float(np.percentile(ytr, 99.5))
    edges = np.linspace(0, max_t + 1e-6, bins + 1)
    tbin = 0.5 * (edges[:-1] + edges[1:])
    btr = np.clip(np.searchsorted(edges, ytr, side="right") - 1, 0, bins - 1).astype(np.int64)
    bva = np.clip(np.searchsorted(edges, yva, side="right") - 1, 0, bins - 1).astype(np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = int(cfg.get("batch_size", 256))
    epochs = int(cfg.get("epochs", 20))
    n_trials = int(cfg.get("n_trials", 10))
    train_ds = FusionDataset(xs[idx["train"]], tv[idx["train"]], tm[idx["train"]], td[idx["train"]], xt[idx["train"]], btr, etr)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)

    def obj(trial):
        hidden = trial.suggest_categorical("hidden", [64, 128, 192, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
        m = FusionNet(xs.shape[1], tv.shape[2], xt.shape[1], hidden, bins, dropout).to(device)
        opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        for _ in range(min(8, epochs)):
            m.train()
            for b in loader:
                s, v, mm, d, x, bi, ev = [t.to(device) for t in b]
                opt.zero_grad()
                logit, _ = m(s, v, mm, d, x)
                loss = nll(logit, bi, ev)
                loss.backward()
                opt.step()
        hz, _ = infer(m, xs[idx["val"]], tv[idx["val"]], tm[idx["val"]], td[idx["val"]], xt[idx["val"]], bs, device)
        sv = np.cumprod(1.0 - hz, axis=1)
        pv = expected_time_from_survival(tbin, sv)
        return compute_point_metrics(yva, eva, pv, -pv)["c_index"]

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    model = FusionNet(xs.shape[1], tv.shape[2], xt.shape[1], int(bp["hidden"]), bins, float(bp["dropout"])).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(bp["lr"]), weight_decay=float(bp["wd"]))
    patience, bad = int(cfg.get("patience", 5)), 0
    best, best_state = -1e9, None
    for _ in range(epochs):
        model.train()
        for b in loader:
            s, v, mm, d, x, bi, ev = [t.to(device) for t in b]
            opt.zero_grad()
            logit, _ = model(s, v, mm, d, x)
            loss = nll(logit, bi, ev)
            loss.backward()
            opt.step()
        hzv, _ = infer(model, xs[idx["val"]], tv[idx["val"]], tm[idx["val"]], td[idx["val"]], xt[idx["val"]], bs, device)
        sv = np.cumprod(1.0 - hzv, axis=1)
        pv = expected_time_from_survival(tbin, sv)
        c = compute_point_metrics(yva, eva, pv, -pv)["c_index"]
        if c > best:
            best, bad = c, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    hz, gates = infer(model, xs, tv, tm, td, xt, bs, device)
    surv_bins = np.cumprod(1.0 - hz, axis=1)
    pred = expected_time_from_survival(tbin, surv_bins)
    return pred, -pred, surv_bins, {"best_params": bp, "best_value": float(study.best_value), "bin_times": tbin.tolist(), "gates_mean": gates.mean(axis=0).tolist(), "state": {k: v.cpu().numpy() for k, v in model.state_dict().items()}}


def maybe_text_strong(data_dir: str, cfg: dict):
    out = os.path.join(data_dir, "X_text_strong.parquet")
    if os.path.exists(out): return {"status": "exists"}
    if not torch.cuda.is_available():
        pd.read_parquet(os.path.join(data_dir, "X_text_fast.parquet")).to_parquet(out, index=False)
        return {"status": "fallback_fast_no_gpu"}
    try:
        from transformers import AutoTokenizer, AutoModel
    except Exception:
        pd.read_parquet(os.path.join(data_dir, "X_text_fast.parquet")).to_parquet(out, index=False)
        return {"status": "fallback_fast_no_transformers"}
    text = pd.read_parquet(os.path.join(data_dir, "text_raw.parquet"))
    tok = AutoTokenizer.from_pretrained(cfg.get("text_model_name", "emilyalsentzer/Bio_ClinicalBERT"))
    model = AutoModel.from_pretrained(cfg.get("text_model_name", "emilyalsentzer/Bio_ClinicalBERT")).cuda().eval()
    bs = int(cfg.get("text_strong_batch_size", 16))
    mx = int(cfg.get("text_max_len", 256))
    vec = []
    with torch.no_grad():
        arr = text["radiology_note_text"].fillna("").astype(str).tolist()
        for i in range(0, len(arr), bs):
            b = tok(arr[i:i + bs], padding=True, truncation=True, max_length=mx, return_tensors="pt").to("cuda")
            o = model(**b).last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            vec.append(o)
    mat = np.vstack(vec)
    df = pd.DataFrame(mat, columns=[f"text_strong__{i:03d}" for i in range(mat.shape[1])])
    df.insert(0, "stay_id", text["stay_id"].astype(int).to_numpy())
    df.to_parquet(out, index=False)
    return {"status": "built"}


def run_e6(bundle, time_grid, out_metrics_dir):
    mets = {e: load_json(os.path.join(out_metrics_dir, f"{e}.json")) for e in ["E3", "E4", "E5"]}
    preds = {}
    survs = {}
    for e in ["E3", "E4", "E5"]:
        pred_e = pd.read_parquet(os.path.join(out_metrics_dir, f"{e}_predictions.parquet")).sort_values("stay_id")["pred_time"].to_numpy(dtype=float)
        fin = np.isfinite(pred_e)
        fallback = float(np.nanmedian(pred_e[fin])) if np.any(fin) else float(np.nanmedian(bundle["duration"]))
        pred_e = np.where(np.isfinite(pred_e), pred_e, fallback)
        cap = float(np.nanpercentile(bundle["duration"], 99.5) * 2.0)
        pred_e = np.clip(pred_e, 0.1, max(cap, 200.0))
        preds[e] = pred_e

        surv_e = np.load(os.path.join(out_metrics_dir, f"{e}_survival.npz"))["surv"].astype(float)
        surv_e = np.where(np.isfinite(surv_e), surv_e, 0.5)
        survs[e] = np.clip(surv_e, 1e-6, 1.0)
    val_scores = np.array([mets["E3"]["val_c_index"], mets["E4"]["val_c_index"], mets["E5"]["val_c_index"]], dtype=float)
    w = np.maximum(val_scores - 0.5, 0.0)
    w[val_scores < 0.6] = 0.0
    if w.sum() <= 0:
        w = np.array([0.0, 0.5, 0.5], dtype=float)
    else:
        w = w / w.sum()
    pred = w[0] * preds["E3"] + w[1] * preds["E4"] + w[2] * preds["E5"]
    surv = w[0] * survs["E3"] + w[1] * survs["E4"] + w[2] * survs["E5"]
    return pred, -pred, surv, {"weights": {"E3": float(w[0]), "E4": float(w[1]), "E5": float(w[2])}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
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

    if args.exp in {"E5", "E6"}:
        save_json(maybe_text_strong(args.data_dir, cfg.get("text", {})), os.path.join(args.out_metrics_dir, "text_strong_status.json"))

    bundle = load_bundle(args.data_dir, args.exp)
    time_grid = make_time_grid(bundle["duration"][bundle["idx"]["train"]], n_grid=100)

    if args.exp == "E0":
        pred, risk, surv, art = run_e0(bundle, time_grid)
    elif args.exp == "E1":
        pred, risk, surv, art = run_e1(bundle, time_grid, cfg.get("E1", {}))
    elif args.exp == "E2":
        pred, risk, surv, art = run_e2(bundle, time_grid, cfg.get("E2", {}))
    elif args.exp == "E3":
        pred, risk, surv, art = run_e3(bundle, time_grid, cfg.get("E3", {}))
    elif args.exp in {"E4", "E5"}:
        mode = "fast" if args.exp == "E4" else "strong"
        pred, risk, surv_bins, art = run_e4_e5(bundle, cfg.get(args.exp, {}), mode)
        bt = np.asarray(art["bin_times"], dtype=float)
        surv = np.vstack([np.interp(time_grid, bt, s, left=1.0, right=s[-1]) for s in surv_bins])
        state = art.pop("state")
        torch.save(state, os.path.join(args.out_models_dir, f"{args.exp}.pt"))
    elif args.exp == "E6":
        pred, risk, surv, art = run_e6(bundle, time_grid, args.out_metrics_dir)
    else:
        raise ValueError(args.exp)

    tree_model = art.pop("__model__", None) if isinstance(art, dict) else None

    met = save_outputs(args.exp, args.out_metrics_dir, bundle, time_grid, pred, risk, np.clip(surv, 1e-6, 1.0), args.n_bootstrap)
    met["artifact"] = art
    save_json(met, os.path.join(args.out_metrics_dir, f"{args.exp}.json"))
    if args.exp in {"E1", "E2", "E3"}:
        with open(os.path.join(args.out_models_dir, f"{args.exp}.pkl"), "wb") as f:
            pickle.dump({"exp": args.exp, "artifact": art}, f)
    if args.exp == "E3" and tree_model is not None:
        tree_model.save_model(os.path.join(args.out_models_dir, "E3_xgb.json"))
    if args.exp == "E2" and tree_model is not None:
        with open(os.path.join(args.out_models_dir, "E2_model.pkl"), "wb") as f:
            pickle.dump(tree_model, f)
    print(f"{args.exp} completed")


if __name__ == "__main__":
    main()
