"""
Score benchmark CSV runs produced by data.benchmark_logger.

Input:
  runs/benchmark_<run_id>_<timestamp>.csv

Outputs:
  - prints run summaries to console
  - writes:
      runs/score_summary_<timestamp>.csv     (one row per run per model)
      runs/score_events_<timestamp>.csv      (optional event/segment latency details)

Metrics:
  - confusion matrix: TP/FP/TN/FN
  - precision, recall, f1, accuracy, FPR
  - detection latency (seconds + samples) for each GT anomaly segment:
      segment = contiguous rows where gt_anomaly==1 per stream_id+node_id
      latency = first detection time - segment start time
      "detection" uses model flag (is_anom_*) or ensemble_3of4

Usage:
  python -m data.score_runs --in runs --glob "benchmark_*.csv" --out runs

Optional:
  python -m data.score_runs --in runs --glob "benchmark_w*_pulse*.csv" --out runs --min-seg-len 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


MODEL_COLUMNS = {
    "rf": "is_anom_rf",
    "lstm": "is_anom_lstm",
    "svm": "is_anom_svm",
    "xgb": "is_anom_xgb",
    "ens3of4": "ensemble_3of4",
}

REQUIRED_COLS = [
    "run_id",
    "topic",
    "stream_id",
    "node_id",
    "payload_ts",
    "gt_anomaly",
    "anomaly_type",
    "vote_sum",
    "ensemble_3of4",
    "is_anom_rf",
    "is_anom_lstm",
    "is_anom_svm",
    "is_anom_xgb",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_int_series(s: pd.Series, default: int = 0) -> pd.Series:
    def _to_int(x):
        try:
            if pd.isna(x):
                return default
            if isinstance(x, bool):
                return int(x)
            if isinstance(x, (int, float)):
                return int(x)
            t = str(x).strip().lower()
            if t in ("true", "t", "yes", "y"):
                return 1
            if t in ("false", "f", "no", "n"):
                return 0
            return int(float(t))
        except Exception:
            return default

    return s.map(_to_int)


def parse_iso_ts_series(s: pd.Series) -> pd.Series:
    # payload_ts is like "2024-02-22T19:36:55Z" or isoformat
    return pd.to_datetime(s, errors="coerce", utc=True)


@dataclass
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return (self.tp / d) if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return (self.tp / d) if d else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        d = self.tp + self.fp + self.tn + self.fn
        return ((self.tp + self.tn) / d) if d else 0.0

    @property
    def fpr(self) -> float:
        d = self.fp + self.tn
        return (self.fp / d) if d else 0.0


def confusion_from(y_true: pd.Series, y_pred: pd.Series) -> Confusion:
    # expects 0/1 ints
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def find_gt_segments(df: pd.DataFrame, key_cols: List[str], min_len: int) -> pd.DataFrame:
    """
    Build contiguous gt_anomaly==1 segments per key (stream/node).
    Returns a dataframe with:
      run_id, stream_id, node_id, seg_id, start_ts, end_ts, start_idx, end_idx, seg_len, anomaly_type
    """
    segments = []

    for key, g in df.groupby(key_cols, sort=False):
        g = g.sort_values("_ts").reset_index(drop=False)  # keep original row index in "index"
        gt = g["gt_anomaly"].astype(int).tolist()

        seg_start = None
        seg_id = 0
        for i, v in enumerate(gt):
            if v == 1 and seg_start is None:
                seg_start = i
            if v == 0 and seg_start is not None:
                seg_end = i - 1
                seg_len = seg_end - seg_start + 1
                if seg_len >= min_len:
                    seg_id += 1
                    start_ts = g.loc[seg_start, "_ts"]
                    end_ts = g.loc[seg_end, "_ts"]
                    a_type = str(g.loc[seg_start, "anomaly_type"] or "unknown")
                    segments.append((*key, seg_id, start_ts, end_ts, seg_start, seg_end, seg_len, a_type))
                seg_start = None

        # close if ended in anomaly
        if seg_start is not None:
            seg_end = len(gt) - 1
            seg_len = seg_end - seg_start + 1
            if seg_len >= min_len:
                seg_id += 1
                start_ts = g.loc[seg_start, "_ts"]
                end_ts = g.loc[seg_end, "_ts"]
                a_type = str(g.loc[seg_start, "anomaly_type"] or "unknown")
                segments.append((*key, seg_id, start_ts, end_ts, seg_start, seg_end, seg_len, a_type))

    out = pd.DataFrame(
        segments,
        columns=key_cols + ["seg_id", "start_ts", "end_ts", "start_i", "end_i", "seg_len", "anomaly_type"],
    )
    return out


def compute_latency_for_segment(
    df_key: pd.DataFrame,
    seg_start_i: int,
    seg_end_i: int,
    pred_col: str,
) -> Tuple[Optional[float], Optional[int]]:
    """
    For a given key-group df (sorted by _ts, reset_index),
    find first predicted anomaly within [seg_start_i, seg_end_i] (or after start until segment end).
    Latency = detected_ts - start_ts
    Returns (latency_seconds, latency_samples) or (None, None) if never detected.
    """
    window = df_key.iloc[seg_start_i : seg_end_i + 1]
    hits = window.index[window[pred_col].astype(int) == 1].tolist()
    if not hits:
        return None, None
    hit_i = hits[0]
    start_ts = df_key.iloc[seg_start_i]["_ts"]
    hit_ts = df_key.loc[hit_i]["_ts"]
    if pd.isna(start_ts) or pd.isna(hit_ts):
        return None, None
    latency_s = float((hit_ts - start_ts).total_seconds())
    latency_samples = int(hit_i - seg_start_i)
    return latency_s, latency_samples


def parse_args():
    ap = argparse.ArgumentParser(description="Score benchmark CSV runs.")
    ap.add_argument("--in", dest="in_dir", default="runs", help="Input dir (default: runs)")
    ap.add_argument("--glob", default="benchmark_*.csv", help="Glob pattern (default: benchmark_*.csv)")
    ap.add_argument("--out", dest="out_dir", default="runs", help="Output dir (default: runs)")
    ap.add_argument("--min-seg-len", type=int, default=3, help="Min contiguous GT length to count as segment (default: 3)")
    ap.add_argument("--no-events", action="store_true", help="Skip per-segment latency output")
    return ap.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"[score] No files matched: {in_dir / args.glob}")

    all_rows = []
    all_events = []

    for fp in files:
        print(f"\n[score] reading: {fp}")
        df = pd.read_csv(fp)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            print(f"[score] SKIP (missing columns): {missing}")
            continue

        # normalize types
        df["gt_anomaly"] = safe_int_series(df["gt_anomaly"], 0)
        for k, col in MODEL_COLUMNS.items():
            if col in df.columns:
                df[col] = safe_int_series(df[col], 0)
        df["_ts"] = parse_iso_ts_series(df["payload_ts"])

        run_id = str(df["run_id"].iloc[0]) if len(df) else fp.stem
        print(f"[score] run_id={run_id} rows={len(df)} streams={df['stream_id'].nunique()} nodes={df['node_id'].nunique()}")

        y_true = df["gt_anomaly"].astype(int)

        # per-model scoring
        for model_name, pred_col in MODEL_COLUMNS.items():
            if pred_col not in df.columns:
                continue
            y_pred = df[pred_col].astype(int)
            cm = confusion_from(y_true, y_pred)

            row = {
                "run_id": run_id,
                "file": fp.name,
                "model": model_name,
                "rows": int(len(df)),
                "streams": int(df["stream_id"].nunique()),
                "nodes": int(df["node_id"].nunique()),
                "tp": cm.tp,
                "fp": cm.fp,
                "tn": cm.tn,
                "fn": cm.fn,
                "precision": cm.precision,
                "recall": cm.recall,
                "f1": cm.f1,
                "accuracy": cm.accuracy,
                "fpr": cm.fpr,
                "gt_rate": float(y_true.mean()) if len(y_true) else 0.0,
                "pred_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
            }
            all_rows.append(row)

        # event/segment latency scoring (optional)
        if not args.no_events:
            key_cols = ["stream_id", "node_id"]
            segs = find_gt_segments(df[df["_ts"].notna()].copy(), key_cols=key_cols, min_len=args.min_seg_len)

            if len(segs) == 0:
                print(f"[score] no GT segments found (min_len={args.min_seg_len})")
            else:
                # compute latency per model per segment
                for (stream_id, node_id), g in df[df["_ts"].notna()].groupby(key_cols, sort=False):
                    g = g.sort_values("_ts").reset_index(drop=True)
                    segs_key = segs[(segs["stream_id"] == stream_id) & (segs["node_id"] == node_id)]
                    if segs_key.empty:
                        continue
                    for _, seg in segs_key.iterrows():
                        for model_name, pred_col in MODEL_COLUMNS.items():
                            if pred_col not in g.columns:
                                continue
                            lat_s, lat_n = compute_latency_for_segment(
                                g,
                                int(seg["start_i"]),
                                int(seg["end_i"]),
                                pred_col=pred_col,
                            )
                            all_events.append({
                                "run_id": run_id,
                                "file": fp.name,
                                "stream_id": stream_id,
                                "node_id": node_id,
                                "seg_id": int(seg["seg_id"]),
                                "anomaly_type": str(seg["anomaly_type"]),
                                "seg_len_samples": int(seg["seg_len"]),
                                "seg_start_ts": str(seg["start_ts"]),
                                "seg_end_ts": str(seg["end_ts"]),
                                "model": model_name,
                                "latency_seconds": lat_s,
                                "latency_samples": lat_n,
                                "detected": 0 if lat_s is None else 1,
                            })

        # console snapshot: show top models quickly
        df_run = pd.DataFrame([r for r in all_rows if r["run_id"] == run_id])
        if not df_run.empty:
            snap = df_run.sort_values(["f1", "recall", "precision"], ascending=False)[
                ["model", "precision", "recall", "f1", "fpr", "tp", "fp", "fn"]
            ]
            print("[score] best-first snapshot:")
            print(snap.to_string(index=False))

    # write outputs
    stamp = utc_stamp()
    summary_path = out_dir / f"score_summary_{stamp}.csv"
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"\n[score] wrote: {summary_path}")

    if not args.no_events:
        events_path = out_dir / f"score_events_{stamp}.csv"
        pd.DataFrame(all_events).to_csv(events_path, index=False)
        print(f"[score] wrote: {events_path}")

    print("[score] done.")


if __name__ == "__main__":
    main()
