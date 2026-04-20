import argparse
import os
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

try:
    from .common import ensure_dir, save_json
except ImportError:
    from common import ensure_dir, save_json


STATIC_FILE = "MIMIC-IV-static(Group Assignment).csv"
TEXT_FILE = "MIMIC-IV-text(Group Assignment).csv"
TS_FILE = "MIMIC-IV-time_series(Group Assignment).csv"


LEAK_COLS = {
    "icu_los_hours",
    "los",
    "outtime",
    "deathtime",
}


def _resolve_raw_files(raw_dir: str) -> Dict[str, str]:
    candidates = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            lower = f.lower()
            if lower.endswith(".csv"):
                candidates.append(os.path.join(root, f))
    mapping = {}
    for p in candidates:
        name = os.path.basename(p)
        if name == STATIC_FILE:
            mapping["static"] = p
        elif name == TEXT_FILE:
            mapping["text"] = p
        elif name == TS_FILE:
            mapping["ts"] = p
    if len(mapping) == 3:
        return mapping

    zips = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(".zip"):
                zips.append(os.path.join(root, f))
    for zp in zips:
        with zipfile.ZipFile(zp, "r") as zf:
            names = zf.namelist()
            has_all = any(STATIC_FILE in n for n in names) and any(
                TEXT_FILE in n for n in names
            ) and any(TS_FILE in n for n in names)
            if has_all:
                extract_dir = os.path.join(raw_dir, "extracted")
                ensure_dir(extract_dir)
                zf.extractall(extract_dir)
                return _resolve_raw_files(extract_dir)
    raise FileNotFoundError(f"Cannot find required CSV files under {raw_dir}")


def build_splits(
    cohort: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    cohort = cohort.copy()
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    idx_train, idx_temp = next(gss1.split(cohort, groups=cohort["subject_id"]))

    temp = cohort.iloc[idx_temp].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    idx_val_rel, idx_test_rel = next(gss2.split(temp, groups=temp["subject_id"]))

    train_df = cohort.iloc[idx_train].reset_index(drop=True)
    val_df = temp.iloc[idx_val_rel].reset_index(drop=True)
    test_df = temp.iloc[idx_test_rel].reset_index(drop=True)

    split_map = {
        "train": train_df["stay_id"].astype(int).tolist(),
        "val": val_df["stay_id"].astype(int).tolist(),
        "test": test_df["stay_id"].astype(int).tolist(),
    }
    return cohort, split_map


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return 0.0
    x2 = x[mask]
    y2 = y[mask]
    if np.std(x2) < 1e-8:
        return 0.0
    return float(np.polyfit(x2, y2, 1)[0])


def create_ts_stat_features(
    ts: pd.DataFrame,
    cohort_ids: np.ndarray,
    ts_vars: List[str],
) -> pd.DataFrame:
    gb = ts.groupby("stay_id", sort=False)
    agg = gb[ts_vars].agg(["min", "max", "mean", "std", "last"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]

    missing = gb[ts_vars].apply(lambda d: d.isna().mean())
    missing.columns = [f"{c}__missing_ratio" for c in missing.columns]

    slopes = []
    for sid, g in gb:
        hour = g["hour_index"].to_numpy(dtype=float)
        row = {"stay_id": sid}
        for c in ts_vars:
            row[f"{c}__slope"] = _safe_slope(hour, g[c].to_numpy(dtype=float))
        slopes.append(row)
    slope_df = pd.DataFrame(slopes).set_index("stay_id")

    out = agg.join(missing, how="left").join(slope_df, how="left")
    out = out.reset_index()
    out = pd.DataFrame({"stay_id": cohort_ids}).merge(out, on="stay_id", how="left")
    for c in out.columns:
        if c != "stay_id":
            out[c] = out[c].astype(float).fillna(0.0)
    return out


def create_ts_seq_features(
    ts: pd.DataFrame,
    cohort_ids: np.ndarray,
    ts_vars: List[str],
    horizon_hours: int,
    out_dir: str,
) -> Dict[str, str]:
    n = len(cohort_ids)
    v = len(ts_vars)
    seq = np.full((n, horizon_hours, v), np.nan, dtype=np.float32)
    stay_to_idx = {int(s): i for i, s in enumerate(cohort_ids)}

    for j, col in enumerate(ts_vars):
        pivot = ts.pivot_table(
            index="stay_id", columns="hour_index", values=col, aggfunc="last"
        )
        pivot = pivot.reindex(index=cohort_ids, columns=list(range(horizon_hours)))
        seq[:, :, j] = pivot.to_numpy(dtype=np.float32)

    mask = np.isfinite(seq).astype(np.float32)
    seq_filled = np.nan_to_num(seq, nan=0.0)

    delta = np.zeros_like(seq_filled, dtype=np.float32)
    delta[:, 0, :] = 1.0
    for t in range(1, horizon_hours):
        delta[:, t, :] = 1.0 + (1.0 - mask[:, t - 1, :]) * delta[:, t - 1, :]

    out_path = os.path.join(out_dir, "X_ts_seq.npz")
    np.savez_compressed(
        out_path,
        stay_id=cohort_ids.astype(np.int64),
        values=seq_filled.astype(np.float32),
        mask=mask.astype(np.float32),
        delta=delta.astype(np.float32),
        ts_vars=np.array(ts_vars),
    )
    return {"ts_seq_path": out_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--horizon_hours", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tfidf_features", type=int, default=40000)
    parser.add_argument("--svd_dim", type=int, default=256)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    file_map = _resolve_raw_files(args.raw_dir)

    static = pd.read_csv(file_map["static"], low_memory=False)
    text = pd.read_csv(file_map["text"], low_memory=False)
    ts = pd.read_csv(file_map["ts"], low_memory=False, na_values=["NULL", "null", ""])

    static["stay_id"] = static["stay_id"].astype(int)
    text["stay_id"] = text["stay_id"].astype(int)
    ts["stay_id"] = ts["stay_id"].astype(int)

    common_ids = sorted(
        set(static["stay_id"]).intersection(text["stay_id"]).intersection(ts["stay_id"])
    )
    static = static[static["stay_id"].isin(common_ids)].copy()
    text = text[text["stay_id"].isin(common_ids)].copy()
    ts = ts[ts["stay_id"].isin(common_ids)].copy()

    static["intime"] = pd.to_datetime(static["intime"])
    static["outtime"] = pd.to_datetime(static["outtime"])
    static["deathtime"] = pd.to_datetime(static["deathtime"], errors="coerce")
    static["icu_death_flag"] = pd.to_numeric(static["icu_death_flag"], errors="coerce").fillna(0).astype(int)
    static["duration"] = pd.to_numeric(static["icu_los_hours"], errors="coerce").astype(float)
    static["event"] = (1 - static["icu_death_flag"]).astype(int)

    cohort = static[["stay_id", "subject_id", "hadm_id", "intime", "duration", "event"]].copy()
    cohort, splits = build_splits(cohort, args.seed)
    split_sets = {k: set(v) for k, v in splits.items()}

    keep_static = []
    for c in static.columns:
        if c in LEAK_COLS:
            continue
        keep_static.append(c)
    static_clean = static[keep_static].copy()

    protected_cols = {
        "stay_id",
        "subject_id",
        "hadm_id",
        "intime",
        "event",
        "duration",
        "icu_death_flag",
    }
    missing_ratio = static_clean.isna().mean()
    drop_cols = [
        c
        for c in static_clean.columns
        if c not in protected_cols and missing_ratio.get(c, 0.0) > 0.999
    ]
    static_clean = static_clean.drop(columns=drop_cols, errors="ignore")

    feature_cols = [c for c in static_clean.columns if c not in protected_cols]
    train_mask = static_clean["stay_id"].isin(split_sets["train"])
    train_df = static_clean.loc[train_mask].copy()

    num_cols = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(static_clean[c]) and c not in {"stay_id"}
    ]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    static_feat = pd.DataFrame({"stay_id": static_clean["stay_id"].astype(int)})

    if num_cols:
        medians = train_df[num_cols].median()
        means = train_df[num_cols].fillna(medians).mean()
        stds = train_df[num_cols].fillna(medians).std().replace(0, 1.0)
        num_filled = static_clean[num_cols].fillna(medians)
        num_scaled = (num_filled - means) / stds
        num_scaled.columns = [f"num__{c}" for c in num_scaled.columns]
        miss = static_clean[num_cols].isna().astype(np.int8)
        miss.columns = [f"num_missing__{c}" for c in miss.columns]
        static_feat = pd.concat([static_feat, num_scaled, miss], axis=1)
    else:
        medians = pd.Series(dtype=float)
        means = pd.Series(dtype=float)
        stds = pd.Series(dtype=float)

    if cat_cols:
        cat_df = static_clean[cat_cols].copy()
        for c in cat_cols:
            train_cats = (
                train_df[c].fillna("Unknown").astype(str).value_counts().index.tolist()
            )
            cat_df[c] = pd.Categorical(
                cat_df[c].fillna("Unknown").astype(str), categories=train_cats
            )
        cat_onehot = pd.get_dummies(cat_df, prefix=[f"cat__{c}" for c in cat_cols], dtype=np.int8)
        static_feat = pd.concat([static_feat, cat_onehot], axis=1)

    static_feat = static_feat.fillna(0)
    static_feat.to_parquet(os.path.join(args.out_dir, "X_static.parquet"), index=False)

    ts["hour_ts"] = pd.to_datetime(ts["hour_ts"])
    intime_map = cohort[["stay_id", "intime"]]
    ts = ts.merge(intime_map, on="stay_id", how="inner")
    ts = ts[(ts["hour_ts"] >= ts["intime"]) & (ts["hour_ts"] < ts["intime"] + pd.Timedelta(hours=args.horizon_hours))].copy()
    ts["hour_index"] = ((ts["hour_ts"] - ts["intime"]) / pd.Timedelta(hours=1)).astype(int)

    ts_vars = [c for c in ts.columns if c not in {"stay_id", "hour_ts", "intime", "hour_index"}]
    for c in ts_vars:
        ts[c] = pd.to_numeric(ts[c], errors="coerce")

    ts_missing = ts[ts_vars].isna().mean()
    ts_vars = [c for c in ts_vars if ts_missing[c] <= 0.999]
    ts = ts[["stay_id", "hour_index"] + ts_vars]

    cohort_ids = cohort["stay_id"].astype(int).to_numpy()
    ts_stat = create_ts_stat_features(ts, cohort_ids=cohort_ids, ts_vars=ts_vars)
    ts_stat.to_parquet(os.path.join(args.out_dir, "X_ts_stat.parquet"), index=False)

    seq_meta = create_ts_seq_features(
        ts=ts,
        cohort_ids=cohort_ids,
        ts_vars=ts_vars,
        horizon_hours=args.horizon_hours,
        out_dir=args.out_dir,
    )

    text_keep = text[["stay_id", "radiology_note_text"]].copy()
    text_keep["radiology_note_text"] = text_keep["radiology_note_text"].fillna("").astype(str)
    text_keep["text_char_len"] = text_keep["radiology_note_text"].str.len().astype(int)
    text_keep.to_parquet(os.path.join(args.out_dir, "text_raw.parquet"), index=False)

    train_ids = split_sets["train"]
    text_train = text_keep[text_keep["stay_id"].isin(train_ids)]["radiology_note_text"].tolist()
    text_all = text_keep["radiology_note_text"].tolist()

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=args.max_tfidf_features,
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_sparse = tfidf.fit_transform(text_train)
    X_all_sparse = tfidf.transform(text_all)

    svd_dim = min(args.svd_dim, max(2, X_train_sparse.shape[1] - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=args.seed)
    X_train_svd = svd.fit_transform(X_train_sparse)
    X_all_svd = svd.transform(X_all_sparse).astype(np.float32)

    text_fast = pd.DataFrame(X_all_svd, columns=[f"text_fast__{i:03d}" for i in range(X_all_svd.shape[1])])
    text_fast.insert(0, "stay_id", text_keep["stay_id"].astype(int).to_numpy())
    text_fast.to_parquet(os.path.join(args.out_dir, "X_text_fast.parquet"), index=False)

    cohort_out = cohort.merge(
        text_keep[["stay_id", "text_char_len"]], on="stay_id", how="left"
    )
    cohort_out.to_parquet(os.path.join(args.out_dir, "cohort.parquet"), index=False)
    save_json(splits, os.path.join(args.out_dir, "splits.json"))

    summary = {
        "n_stays": int(len(cohort_out)),
        "n_train": int(len(splits["train"])),
        "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "horizon_hours": int(args.horizon_hours),
        "n_static_features": int(static_feat.shape[1] - 1),
        "n_ts_stat_features": int(ts_stat.shape[1] - 1),
        "n_ts_seq_vars": int(len(ts_vars)),
        "n_text_fast_dim": int(X_all_svd.shape[1]),
        "dropped_static_cols_missing_gt_99_9": drop_cols,
        "ts_seq_path": seq_meta["ts_seq_path"],
    }
    save_json(summary, os.path.join(args.out_dir, "preprocess_summary.json"))

    artifact = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "drop_cols": drop_cols,
        "ts_vars": ts_vars,
        "tfidf_vocab_size": int(len(tfidf.vocabulary_)),
        "svd_dim": int(X_all_svd.shape[1]),
    }
    save_json(artifact, os.path.join(args.out_dir, "feature_artifacts.json"))

    print("Preprocessing completed.")
    print(summary)


if __name__ == "__main__":
    main()
