"""
Replay normalized CSV rows to MQTT topics at a fixed rate,
with enriching each row with moving average & moving std.

Expected input CSV columns (normalized):
  timestamp,node_id,type,power_kw,voltage,frequency_hz

Publishes JSON to:
  {topic_prefix}/{node_id}/telemetry

Examples:
  # 4 streams, per-stream offsets auto, moving stats on frequency
  python -m data.replay_mqtt --csv data/normalized.csv --host localhost --rate 20 --loop \
      --streams 4 --ma-window 200 --ma-signal frequency_hz

  # Same, but inject pulse anomalies
  python -m data.replay_mqtt --csv data/normalized.csv --host localhost --rate 20 --loop \
      --streams 4 --ma-window 200 --ma-signal frequency_hz \
      --anomaly-mode pulse --pulse-period 300 --pulse-duration 20 --pulse-mag 0.08

  # Random anomalies (Bernoulli per message)
  python -m data.replay_mqtt --csv data/normalized.csv --host localhost --rate 20 --loop \
      --streams 4 --ma-window 200 --ma-signal frequency_hz \
      --anomaly-mode random --random-prob 0.01 --random-mag 0.06
"""

from __future__ import annotations

import argparse
import json
import time
import threading
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd

from comm.mqtt_client import MQTTClient
from .moving_avg import Node_MA
from .data_adjust import data_adjust
from .ml_models import RF_Classifier, SVM_Classifier, LSTM_Classifier, XGB_Classifier


REQUIRED_COLS = ["timestamp", "node_id", "power_kw"]


def parse_args():
    ap = argparse.ArgumentParser(description="Replay normalized CSV to MQTT (multi-stream).")
    ap.add_argument("--csv", default="data/normalized.csv", help="Path to normalized CSV")
    ap.add_argument("--host", default="localhost", help="MQTT broker host")
    ap.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    ap.add_argument("--topic-prefix", dest="topic_prefix", default="derms", help="Topic prefix (default: derms)")
    ap.add_argument("--rate", type=float, default=10.0, help="Messages per second PER STREAM")
    ap.add_argument("--limit", type=int, default=None, help="Max rows to send then exit (per stream)")
    ap.add_argument("--chunksize", type=int, default=10_000, help="CSV read chunk size")
    ap.add_argument("--loop", action="store_true", help="Loop over the CSV forever")

    # Multi-stream
    ap.add_argument("--streams", type=int, default=1, help="Number of parallel streams to publish (default: 1)")
    ap.add_argument(
        "--offset-hours",
        type=str,
        default="",
        help="Comma-separated timestamp offsets per stream in hours (e.g., 0,24,48,72). "
             "If omitted, defaults to 24h increments.",
    )

    # Moving statistics
    ap.add_argument("--ma-window", type=int, default=None, help="Enable moving stats with this window size")
    ap.add_argument("--ma-signal", default="frequency_hz", help="Column to smooth (e.g., frequency_hz or power_kw)")
    ap.add_argument("--ma-field", default=None, help="Output field for moving average (default: moving_avg_<signal>)")
    ap.add_argument("--std-field", default=None, help="Output field for moving std (default: moving_std_<signal>)")
    ap.add_argument("--std-ddof", type=int, default=1, help="ddof for sample std (default 1)")

    # Anomaly injection
    ap.add_argument("--anomaly-mode", choices=["none", "random", "pulse", "both", "step", "drift"], default="none")
    ap.add_argument("--random-prob", type=float, default=0.0, help="Random anomaly probability per message (0..1)")
    ap.add_argument("--random-mag", type=float, default=0.05, help="Magnitude added to ma_signal when random anomaly fires")
    ap.add_argument("--pulse-period", type=int, default=300, help="Pulse anomaly period in messages")
    ap.add_argument("--pulse-duration", type=int, default=20, help="Pulse anomaly duration in messages")
    ap.add_argument("--pulse-mag", type=float, default=0.08, help="Pulse anomaly magnitude added to ma_signal")
    ap.add_argument("--step-start", type=int, default=200, help="Step anomaly start index in messages")
    ap.add_argument("--step-duration", type=int, default=80, help="Step anomaly duration in messages")
    ap.add_argument("--step-mag", type=float, default=0.08, help="Step anomaly magnitude added to ma_signal")
    ap.add_argument("--drift-start", type=int, default=200, help="Drift anomaly start index in messages")
    ap.add_argument("--drift-duration", type=int, default=300, help="Drift anomaly duration in messages")
    ap.add_argument("--drift-mag", type=float, default=0.12, help="Max drift anomaly magnitude added to ma_signal")

    return ap.parse_args()


def _validate_columns(df: pd.DataFrame, need_ma: bool, ma_signal: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[replay] CSV missing required columns: {missing}")
    if need_ma and ma_signal not in df.columns:
        raise ValueError(f"[replay] --ma-signal '{ma_signal}' not found in CSV columns.")


def _parse_offsets(streams: int, offset_hours_arg: str) -> List[int]:
    if offset_hours_arg.strip():
        parts = [p.strip() for p in offset_hours_arg.split(",") if p.strip() != ""]
        offsets = [int(float(p)) for p in parts]
        if len(offsets) != streams:
            raise ValueError(f"[replay] --offset-hours must provide exactly {streams} values, got {len(offsets)}")
        return offsets
    # default: 0,24,48,...
    return [24 * i for i in range(streams)]


def _apply_timestamp_offset(payload: Dict[str, Any], offset_hours: int):
    if offset_hours == 0:
        return
    if "timestamp" not in payload:
        return
    try:
        ts = pd.Timestamp(payload["timestamp"])
        payload["timestamp"] = str(ts + pd.Timedelta(hours=offset_hours))
    except Exception:
        # If timestamp parse fails, leave it unchanged (viz uses ingest-time anyway)
        return


def _inject_anomaly(
    idx_msg: int,
    base_value: float,
    mode: str,
    rnd_prob: float,
    rnd_mag: float,
    pulse_period: int,
    pulse_duration: int,
    pulse_mag: float,
    step_start: int,
    step_duration: int,
    step_mag: float,
    drift_start: int,
    drift_duration: int,
    drift_mag: float,
) -> Tuple[float, int, str, float]:
    """
    Returns (value, gt_anomaly, anomaly_type, anomaly_mag)
    """
    gt = 0
    a_type = "none"
    a_mag = 0.0
    v = base_value

    do_random = mode in ("random", "both") and rnd_prob > 0.0
    do_pulse = mode in ("pulse", "both") and pulse_period > 0 and pulse_duration > 0
    do_step = mode == "step" and step_duration > 0
    do_drift = mode == "drift" and drift_duration > 0

    # step: constant offset for [start, start+duration)
    if do_step:
        step_end = step_start + step_duration
        if step_start <= idx_msg < step_end:
            gt = 1
            a_type = "step"
            a_mag = float(step_mag)
            v = base_value + a_mag

    # drift: linear ramp from 0..drift_mag across [start, start+duration)
    if do_drift:
        drift_end = drift_start + drift_duration
        if drift_start <= idx_msg < drift_end:
            if drift_duration == 1:
                delta = float(drift_mag)
            else:
                progress = (idx_msg - drift_start) / (drift_duration - 1)
                delta = float(drift_mag) * progress
            gt = 1
            a_type = "drift"
            a_mag = delta
            v = base_value + delta

    # pulse: on for [0..duration) each period
    if do_pulse:
        phase = idx_msg % pulse_period
        if phase < pulse_duration:
            gt = 1
            a_type = "pulse"
            a_mag = float(pulse_mag)
            v = base_value + a_mag

    # random can override "none" or stack on top of pulse; keep it simple:
    if do_random and random.random() < rnd_prob:
        gt = 1
        # if already pulse, mark as both-ish but keep string simple:
        a_type = "random" if a_type == "none" else a_type
        a_mag = a_mag + float(rnd_mag)
        v = v + float(rnd_mag)

    return v, gt, a_type, float(a_mag)


def _publish_dataframe(
    df: pd.DataFrame,
    mqc: MQTTClient,
    topic_prefix: str,
    rate: float,
    limit: Optional[int],
    ma: Optional[Node_MA],
    ma_signal: str,
    ma_field: str,
    std_field: str,
    adjuster,
    models,
    stream_id: str,
    time_offset_hours: int,
    anomaly_cfg: Dict[str, Any],
):
    sleep = 1.0 / max(rate, 0.001)
    sent = 0
    printed = 0
    msg_idx = 0

    for _, row in df.iterrows():
        node = str(row["node_id"])
        topic = f"{topic_prefix}/{stream_id}/{node}/telemetry"

        payload = {k: v for k, v in row.dropna().to_dict().items()}

        payload["stream_id"] = stream_id
        _apply_timestamp_offset(payload, time_offset_hours)

        # base signal (e.g., frequency_hz)
        base = row.get(ma_signal)
        try:
            base = float(base)
        except Exception:
            base = 0.0

        # small natural adjuster (existing)
        adjusted_value = base + adjuster.adjust()

        # anomaly injection on adjusted value
        adjusted_value, gt_anom, anom_type, anom_mag = _inject_anomaly(
            idx_msg=msg_idx,
            base_value=adjusted_value,
            mode=anomaly_cfg["mode"],
            rnd_prob=anomaly_cfg["random_prob"],
            rnd_mag=anomaly_cfg["random_mag"],
            pulse_period=anomaly_cfg["pulse_period"],
            pulse_duration=anomaly_cfg["pulse_duration"],
            pulse_mag=anomaly_cfg["pulse_mag"],
            step_start=anomaly_cfg["step_start"],
            step_duration=anomaly_cfg["step_duration"],
            step_mag=anomaly_cfg["step_mag"],
            drift_start=anomaly_cfg["drift_start"],
            drift_duration=anomaly_cfg["drift_duration"],
            drift_mag=anomaly_cfg["drift_mag"],
        )
        msg_idx += 1

        payload["Hz_adjusted"] = adjusted_value
        payload["gt_anomaly"] = int(gt_anom)
        payload["anomaly_type"] = anom_type
        payload["anomaly_mag"] = float(anom_mag)

        # moving stats + model flags
        if ma is not None:
            avg = ma.update(adjusted_value)
            std = ma.std()
            payload[ma_field] = float(avg)
            payload[std_field] = float(std)

            ma60_val = ma.average(60)
            ma120_val = ma.average(120)
            ma180_val = ma.average(180)
            ma240_val = ma.average(240)

            feats = [adjusted_value, ma60_val, ma120_val, ma180_val, ma240_val, ma.std(60), ma.std(120)]
            payload["is_anom_rf"] = int(models["rf"].classify(feats))
            payload["is_anom_lstm"] = int(models["lstm"].classify(feats))
            payload["is_anom_svm"] = int(models["svm"].classify(feats))
            payload["is_anom_xgb"] = int(models["xgb"].classify(feats))
        else:
            # keep fields present (helps UI consistency)
            payload.setdefault("is_anom_rf", 0)
            payload.setdefault("is_anom_lstm", 0)
            payload.setdefault("is_anom_svm", 0)
            payload.setdefault("is_anom_xgb", 0)

        if printed < 5:
            preview = json.dumps(payload, default=str)
            if len(preview) > 260:
                preview = preview[:260] + "...}"
            print(f"[replay] topic={topic} payload={preview}")
            printed += 1

        mqc.publish(topic, payload)
        sent += 1

        if sent % 1000 == 0:
            print(f"[replay|{stream_id}] sent={sent}")

        if limit and sent >= limit:
            print(f"[replay|{stream_id}] sent {sent} messages; exiting")
            return sent

        time.sleep(sleep)

    return sent


def _one_pass(
    csv_path: Path,
    chunksize: int,
    mqc: MQTTClient,
    topic_prefix: str,
    rate: float,
    limit: Optional[int],
    ma: Optional[Node_MA],
    ma_signal: str,
    ma_field: str,
    std_field: str,
    models,
    stream_id: str,
    time_offset_hours: int,
    anomaly_cfg: Dict[str, Any],
):
    g = data_adjust(magnitude=0.05, method="gaussian", period=300)

    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        _validate_columns(chunk, need_ma=ma is not None, ma_signal=ma_signal)
        total += _publish_dataframe(
            chunk,
            mqc,
            topic_prefix,
            rate,
            None if limit is None else max(0, limit - total),
            ma,
            ma_signal,
            ma_field,
            std_field,
            g,
            models,
            stream_id=stream_id,
            time_offset_hours=time_offset_hours,
            anomaly_cfg=anomaly_cfg,
        )
        if limit is not None and total >= limit:
            break
    return total


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"[replay] CSV not found: {csv_path.resolve()}")

    streams = max(1, int(args.streams))
    offsets = _parse_offsets(streams, args.offset_hours)

    use_ma = args.ma_window is not None and args.ma_window > 0
    ma_field = args.ma_field or (f"moving_avg_{args.ma_signal}" if use_ma else "")
    std_field = args.std_field or (f"moving_std_{args.ma_signal}" if use_ma else "")

    anomaly_cfg = {
        "mode": args.anomaly_mode,
        "random_prob": float(args.random_prob),
        "random_mag": float(args.random_mag),
        "pulse_period": int(args.pulse_period),
        "pulse_duration": int(args.pulse_duration),
        "pulse_mag": float(args.pulse_mag),
        "step_start": int(args.step_start),
        "step_duration": int(args.step_duration),
        "step_mag": float(args.step_mag),
        "drift_start": int(args.drift_start),
        "drift_duration": int(args.drift_duration),
        "drift_mag": float(args.drift_mag),
    }

    print(
        f"[replay] starting\n"
        f"  csv          = {csv_path}\n"
        f"  host:port    = {args.host}:{args.port}\n"
        f"  topic_prefix = {args.topic_prefix}\n"
        f"  streams      = {streams}\n"
        f"  offsets(h)   = {offsets}\n"
        f"  rate         = {args.rate} msg/s (per stream)\n"
        f"  limit        = {args.limit} (rows per stream)\n"
        f"  chunksize    = {args.chunksize}\n"
        f"  loop         = {args.loop}\n"
        f"  moving_stats = {'ON' if use_ma else 'OFF'}"
        f"{'' if not use_ma else f' (signal={args.ma_signal}, window={args.ma_window}, avg_field={ma_field}, std_field={std_field})'}\n"
        f"  anomaly      = {anomaly_cfg}\n"
    )

    def run_stream(i: int, offset_h: int):
        stream_id = f"stream_{i}"

        ma = Node_MA(n=args.ma_window, ddof=args.std_ddof) if use_ma else None

        # NOTE: This matches your current approach: each stream loads models independently.
        # If startup becomes slow, you can later move models outside and share across threads.
        models = {
            "rf": RF_Classifier(),
            "lstm": LSTM_Classifier(),
            "svm": SVM_Classifier(),
            "xgb": XGB_Classifier(),
        }

        mqc = MQTTClient(client_id=f"replayer_{stream_id}", host=args.host, port=args.port)

        if args.loop:
            pass_idx = 0
            while True:
                pass_idx += 1
                print(f"[replay] pass #{pass_idx} {stream_id} (offset={offset_h}h)")
                _one_pass(
                    csv_path,
                    args.chunksize,
                    mqc,
                    args.topic_prefix,
                    args.rate,
                    args.limit,
                    ma,
                    args.ma_signal,
                    ma_field,
                    std_field,
                    models,
                    stream_id=stream_id,
                    time_offset_hours=offset_h,
                    anomaly_cfg=anomaly_cfg,
                )
                if args.limit is not None:
                    break
                time.sleep(0.5)
        else:
            print(f"[replay] {stream_id} offset=+{offset_h}h")
            _one_pass(
                csv_path,
                args.chunksize,
                mqc,
                args.topic_prefix,
                args.rate,
                args.limit,
                ma,
                args.ma_signal,
                ma_field,
                std_field,
                models,
                stream_id=stream_id,
                time_offset_hours=offset_h,
                anomaly_cfg=anomaly_cfg,
            )

    threads = []
    try:
        for idx in range(streams):
            t = threading.Thread(target=run_stream, args=(idx + 1, offsets[idx]), daemon=True)
            threads.append(t)
            t.start()

        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[replay] interrupted by user")
    finally:
        print("[replay] done.")


if __name__ == "__main__":
    main()
