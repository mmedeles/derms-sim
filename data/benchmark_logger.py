"""
Benchmark Logger: subscribes to MQTT and writes telemetry + detection outputs to CSV.

Designed for benchmarking:
- Moving average window sweep: 10/15/20/30 (run separate replayer commands)
- Anomaly types: pulse vs random (via replayer flags)
- Metrics: false positives, false negatives, precision/recall, detection latency proxies, vote behavior

Subscribes:
  derms/#   (default)

Expects telemetry payloads like:
  topic: derms/stream_1/inverter_1/telemetry
  payload JSON includes:
    timestamp, node_id, stream_id,
    gt_anomaly (0/1), anomaly_type, anomaly_mag,
    is_anom_rf/is_anom_lstm/is_anom_svm/is_anom_xgb (0/1)

Writes CSV rows with:
  ingest_ts_utc, run_id, topic, stream_id, node_id,
  payload_ts, power_kw, voltage, frequency_hz, Hz_adjusted,
  gt_anomaly, anomaly_type, anomaly_mag,
  is_anom_rf, is_anom_lstm, is_anom_svm, is_anom_xgb,
  vote_sum, ensemble_3of4

Usage:
  python -m data.benchmark_logger --host localhost --port 1883 --topic 'derms/#' --out runs

Then run your replayer in another terminal.

Stop with Ctrl+C. CSV is flushed frequently for safety.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import paho.mqtt.client as mqtt


TOPIC_RE = re.compile(r"^(?P<prefix>[^/]+)/(?P<stream>[^/]+)/(?P<node>[^/]+)/(?P<kind>telemetry|anomaly)$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).strip()
        if s == "":
            return default
        if s.lower() in ("true", "t", "yes", "y"):
            return 1
        if s.lower() in ("false", "f", "no", "n"):
            return 0
        return int(float(s))
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def parse_topic(topic: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (stream_id, node_id, kind) or (None, None, None) if not matching.
    """
    m = TOPIC_RE.match(topic or "")
    if not m:
        return None, None, None
    return m.group("stream"), m.group("node"), m.group("kind")


@dataclass
class Row:
    ingest_ts_utc: str
    run_id: str
    topic: str
    stream_id: str
    node_id: str
    payload_ts: str

    power_kw: float
    voltage: float
    frequency_hz: float
    hz_adjusted: float

    gt_anomaly: int
    anomaly_type: str
    anomaly_mag: float

    is_anom_rf: int
    is_anom_lstm: int
    is_anom_svm: int
    is_anom_xgb: int

    vote_sum: int
    ensemble_3of4: int


CSV_FIELDS = [
    "ingest_ts_utc",
    "run_id",
    "topic",
    "stream_id",
    "node_id",
    "payload_ts",
    "power_kw",
    "voltage",
    "frequency_hz",
    "Hz_adjusted",
    "gt_anomaly",
    "anomaly_type",
    "anomaly_mag",
    "is_anom_rf",
    "is_anom_lstm",
    "is_anom_svm",
    "is_anom_xgb",
    "vote_sum",
    "ensemble_3of4",
]


class BenchmarkLogger:
    def __init__(self, out_dir: Path, run_id: str, flush_every: int = 50):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.out_dir / f"benchmark_{run_id}_{ts}.csv"
        self.run_id = run_id

        self.flush_every = max(1, int(flush_every))
        self._n = 0

        self._fp = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._fp.flush()

        print(f"[bench] writing to: {self.csv_path}")

    def close(self):
        try:
            self._fp.flush()
        except Exception:
            pass
        try:
            self._fp.close()
        except Exception:
            pass

    def write_row(self, r: Row):
        self._writer.writerow({
            "ingest_ts_utc": r.ingest_ts_utc,
            "run_id": r.run_id,
            "topic": r.topic,
            "stream_id": r.stream_id,
            "node_id": r.node_id,
            "payload_ts": r.payload_ts,
            "power_kw": r.power_kw,
            "voltage": r.voltage,
            "frequency_hz": r.frequency_hz,
            "Hz_adjusted": r.hz_adjusted,
            "gt_anomaly": r.gt_anomaly,
            "anomaly_type": r.anomaly_type,
            "anomaly_mag": r.anomaly_mag,
            "is_anom_rf": r.is_anom_rf,
            "is_anom_lstm": r.is_anom_lstm,
            "is_anom_svm": r.is_anom_svm,
            "is_anom_xgb": r.is_anom_xgb,
            "vote_sum": r.vote_sum,
            "ensemble_3of4": r.ensemble_3of4,
        })

        self._n += 1
        if (self._n % self.flush_every) == 0:
            self._fp.flush()
            # lightweight progress line
            print(f"[bench] rows={self._n}", end="\r", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Subscribe to MQTT and log benchmark rows to CSV.")
    ap.add_argument("--host", default="localhost", help="MQTT host")
    ap.add_argument("--port", type=int, default=1883, help="MQTT port")
    ap.add_argument("--topic", default="derms/#", help="Topic filter (default: derms/#)")
    ap.add_argument("--out", default="runs", help="Output directory for CSV files (default: runs)")
    ap.add_argument("--run-id", default="", help="Run identifier string (recommended: window/anom/interval tags)")
    ap.add_argument("--flush-every", type=int, default=50, help="Flush CSV every N rows (default: 50)")
    ap.add_argument("--qos", type=int, default=0, choices=[0, 1, 2], help="MQTT QoS (default: 0)")
    return ap.parse_args()


def main():
    args = parse_args()

    run_id = args.run_id.strip()
    if not run_id:
        # sensible default if user forgets
        run_id = "run"

    logger = BenchmarkLogger(out_dir=Path(args.out), run_id=run_id, flush_every=args.flush_every)

    def on_connect(client, userdata, flags, rc):
        print(f"[bench] CONNECTED rc={rc} host={args.host}:{args.port}")
        client.subscribe(args.topic, qos=args.qos)
        print(f"[bench] SUBSCRIBED {args.topic} (qos={args.qos})")

    def on_message(client, userdata, msg):
        topic = msg.topic or ""
        stream_id, node_id, kind = parse_topic(topic)

        # Only log telemetry for Phase 1 (anomaly topic is optional)
        if kind != "telemetry" or stream_id is None or node_id is None:
            return

        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return

        ingest_ts = utc_now_iso()

        # prefer payload fields if present
        p_stream = str(payload.get("stream_id") or stream_id)
        p_node = str(payload.get("node_id") or node_id)
        payload_ts = str(payload.get("timestamp") or "")

        power_kw = safe_float(payload.get("power_kw"), 0.0)
        voltage = safe_float(payload.get("voltage"), 0.0)
        freq_hz = safe_float(payload.get("frequency_hz"), 0.0)
        hz_adj = safe_float(payload.get("Hz_adjusted"), 0.0)

        gt = safe_int(payload.get("gt_anomaly"), 0)
        a_type = str(payload.get("anomaly_type") or "none")
        a_mag = safe_float(payload.get("anomaly_mag"), 0.0)

        rf = safe_int(payload.get("is_anom_rf"), 0)
        lstm = safe_int(payload.get("is_anom_lstm"), 0)
        svm = safe_int(payload.get("is_anom_svm"), 0)
        xgb = safe_int(payload.get("is_anom_xgb"), 0)

        vote_sum = int(rf + lstm + svm + xgb)
        ensemble = 1 if vote_sum >= 3 else 0

        row = Row(
            ingest_ts_utc=ingest_ts,
            run_id=logger.run_id,
            topic=topic,
            stream_id=p_stream,
            node_id=p_node,
            payload_ts=payload_ts,
            power_kw=power_kw,
            voltage=voltage,
            frequency_hz=freq_hz,
            hz_adjusted=hz_adj,
            gt_anomaly=gt,
            anomaly_type=a_type,
            anomaly_mag=a_mag,
            is_anom_rf=rf,
            is_anom_lstm=lstm,
            is_anom_svm=svm,
            is_anom_xgb=xgb,
            vote_sum=vote_sum,
            ensemble_3of4=ensemble,
        )
        logger.write_row(row)

    client = mqtt.Client(protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.host, args.port, 60)
    except Exception as e:
        logger.close()
        raise RuntimeError(f"[bench] Could not connect to MQTT {args.host}:{args.port}: {e}")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[bench] stopped by user")
    finally:
        logger.close()
        print(f"[bench] saved: {logger.csv_path}")


if __name__ == "__main__":
    main()
