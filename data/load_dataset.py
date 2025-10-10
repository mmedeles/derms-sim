"""
Robust, chunked normalization for large inverter CSVs.

Output columns:
  timestamp,node_id,type,power_kw,voltage,frequency_hz
  for now...
"""

import argparse
import pandas as pd
import yaml
from pathlib import Path

# ------- helpers

def to_iso_utc(series, parse_format=None, force_iso=True):
    # Parse timestamps; assume timezone-aware if present, else treat as naive UTC
    dt = pd.to_datetime(series, format=parse_format, utc=True, errors="coerce")
    if force_iso:
        return dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.astype(str)

def pick_col(df, preferred, aliases, default=None, required=False, note=""):
    """
    Return a Series for the first existing column among:
      - preferred (string or None)
      - aliases (list of strings)
    Matching is case-insensitive and ignores surrounding spaces.
    If not found: return a Series filled with default (or raise if required).
    """
    cols = {c.lower().strip(): c for c in df.columns}
    candidates = []
    if preferred:
      candidates.append(preferred)
    candidates.extend(aliases or [])
    for name in candidates:
        key = name.lower().strip()
        if key in cols:
            return pd.to_numeric(df[cols[key]], errors="ignore")  # numeric where possible
    if required:
        raise KeyError(f"Required column not found ({note}). Tried: {candidates}. CSV has: {list(df.columns)[:25]}...")
    # fallback default
    return pd.Series([default] * len(df))

def normalize_chunk(df, cfg, print_cols_once=[True]):
    # Print columns once for debugging
    if print_cols_once[0]:
        print_cols_once[0] = False
        print("[normalize] First chunk columns:", list(df.columns))

    m = cfg.get("map", {})
    defaults_cfg = cfg.get("defaults", {})

    # Timestamp
    ts_series = None
    ts_pref = m.get("timestamp")
    # common timestamp aliases seen in inverter exports
    ts_aliases = ["time", "timestamp", "datetime", "date_time", "datetimestamp", "datetimestampapha"]
    ts_series = pick_col(df, ts_pref, ts_aliases, required=True, note="timestamp")

    # Power: watts -> kW
    p_pref = m.get("power_w") or m.get("power") or "W"
    p_aliases = ["w", "watts", "activepower", "p", "power(w)", "real_power_w", "realpower"]
    p_series = pick_col(df, p_pref, p_aliases, default=None, required=True, note="power (W)")
    power_kw = pd.to_numeric(p_series, errors="coerce") / 1000.0

    # Voltage
    v_pref = m.get("voltage") or "PhVphA"
    v_aliases = ["phvpha", "voltage", "volts", "v", "vpha", "grid_voltage"]
    voltage = pd.to_numeric(pick_col(df, v_pref, v_aliases, default=120.0), errors="coerce")

    # Frequency (Hz)
    f_pref = m.get("frequency_hz") or "TmsHz"
    f_aliases = ["tmshz", "hz", "frequency", "gridhz", "freq", "grid_frequency"]
    frequency_hz = pd.to_numeric(pick_col(df, f_pref, f_aliases, default=60.0), errors="coerce")

    # Node/type defaults
    node_id = defaults_cfg.get("node_id", "inverter_1")
    der_type = defaults_cfg.get("type", "solar")

    # Timestamp formatting
    tfmt = (cfg.get("time") or {}).get("parse_format", None)
    timestamp = to_iso_utc(ts_series, parse_format=tfmt, force_iso=True)

    out = pd.DataFrame({
        "timestamp": timestamp,
        "node_id": node_id,
        "type": der_type,
        "power_kw": power_kw,
        "voltage": voltage,
        "frequency_hz": frequency_hz
    })

    # Drop rows missing essential fields
    out = out.dropna(subset=["timestamp", "power_kw"])
    return out

# ------- main

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to LARGE CSV")
    ap.add_argument("--config", default="configs/dataset_map.yml")
    ap.add_argument("--output", default=None, help="Override output path from config")
    ap.add_argument("--chunksize", type=int, default=100_000)
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    out_path = Path(args.output or cfg.get("output", "data/normalized.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = True
    for chunk in pd.read_csv(args.input, chunksize=args.chunksize):
        norm = normalize_chunk(chunk, cfg)
        norm.to_csv(out_path, mode=("w" if first else "a"), index=False, header=first)
        first = False

    print(f"[normalize] Wrote normalized CSV → {out_path}")

if __name__ == "__main__":
    main()
