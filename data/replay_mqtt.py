"""
Replay normalized CSV rows to MQTT topics at a fixed rate,
enriching each row with moving average & moving std.

Expected input CSV columns (normalized):
  timestamp,node_id,type,power_kw,voltage,frequency_hz

Publishes JSON to:
  {topic_prefix}/stream_<id>/{node_id}/telemetry

Example:
  python -m data.replay_mqtt --csv data/normalized.csv --host localhost --port 1883 \
      --rate 20 --loop --streams 4 --ma-window 200 --ma-signal frequency_hz
"""

from __future__ import annotations

import argparse
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import paho.mqtt.client as mqtt

from .moving_avg import Node_MA
from .data_adjust import data_adjust
from .ml_models import RF_Classifier, SVM_Classifier, LSTM_Classifier, XGB_Classifier


REQUIRED_COLS = ["timestamp", "node_id", "power_kw"]


def parse_args():
    ap = argparse.ArgumentParser(description="Replay normalized CSV to MQTT (multi-stream).")
    ap.add_argument("--csv", default="data/normalized.csv", help="Path to normalized CSV")
    ap.add_argument("--host", default="localhost", help="MQTT broker host")
    ap.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    ap.add_argument("--topic-prefix", dest="topic_prefix", default="derms", help="Topic prefix")
    ap.add_argument("--rate", type=float, default=10.0, help="Messages per second (per stream)")
    ap.add_argument("--limit", type=int, default=None, help="Max rows per stream to send then exit")
    ap.add_argument("--chunksize", type=int, default=10_000, help="CSV read chunk size")
    ap.add_argument("--loop", action="store_true", help="Loop over the CSV forever")

    ap.add_argument("--streams", type=int, default=1, help="Number of independent streams (default: 1)")

    # Moving statistics
    ap.add_argument("--ma-window", type=int, default=None, help="Enable moving stats with this window size")
    ap.add_argument("--ma-signal", default="frequency_hz", help="Column to smooth (e.g., frequency_hz)")
    ap.add_argument("--ma-field", default=None, help="Output field for moving average (default: moving_avg_<signal>)")
    ap.add_argument("--std-field", default=None, help="Output field for moving std (default: moving_std_<signal>)")
    ap.add_argument("--std-ddof", type=int, default=1, help="ddof for sample std (default 1)")
    return ap.parse_args()


def _validate_columns(df: pd.DataFrame, need_ma: bool, ma_signal: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[replay] CSV missing required columns: {missing}")
    if need_ma and ma_signal not in df.columns:
        raise ValueError(f"[replay] --ma-signal '{ma_signal}' not found in CSV columns.")


def _stream_label(stream_id: int) -> str:
    return f"stream_{stream_id}"


def _topic(prefix: str, stream_label: str, node_id: str) -> str:
    return f"{prefix}/{stream_label}/{node_id}/telemetry"


def _ensure_ma(
    ma_by_key: Dict[Tuple[int, str], Node_MA],
    stream_id: int,
    node_id: str,
    ma_window: int,
    std_ddof: int,
) -> Node_MA:
    key = (stream_id, node_id)
    if key not in ma_by_key:
        ma_by_key[key] = Node_MA(n=ma_window, ddof=std_ddof)
    return ma_by_key[key]


def _connect_mqtt(host: str, port: int, client_id: str = "replayer", timeout_s: int = 8) -> mqtt.Client:
    connected = threading.Event()
    failed = {"err": None}

    def on_connect(c, u, flags, rc):
        # rc==0 success
        if rc == 0:
            print(f"[mqtt] CONNECTED to {host}:{port} (rc={rc})")
            connected.set()
        else:
            failed["err"] = f"connect failed rc={rc}"
            print(f"[mqtt] ERROR: {failed['err']}")
            connected.set()

    def on_disconnect(c, u, rc):
        print(f"[mqtt] DISCONNECTED rc={rc}")

    def on_log(c, u, level, buf):
        # Uncomment if you need deep diagnostics:
        # print("[mqtt-log]", buf)
        pass

    client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    # client.on_log = on_log

    # async connect + start loop BEFORE we publish anything
    client.connect_async(host, port, keepalive=60)
    client.loop_start()

    if not connected.wait(timeout=timeout_s):
        client.loop_stop()
        raise RuntimeError(f"[mqtt] ERROR: timed out waiting for connection to {host}:{port}")

    if failed["err"] is not None:
        client.loop_stop()
        raise RuntimeError(f"[mqtt] ERROR: {failed['err']} (host={host} port={port})")

    return client


def _publish_rows(
    df: pd.DataFrame,
    client: mqtt.Client,
    topic_prefix: str,
    streams: int,
    rate: float,
    limit_rows_per_stream: Optional[int],
    use_ma: bool,
    ma_by_key: Dict[Tuple[int, str], Node_MA],
    ma_window: int,
    std_ddof: int,
    ma_signal: str,
    ma_field: str,
    std_field: str,
    adjuster_by_stream: Dict[int, object],
    models,
) -> int:
    sleep = 1.0 / max(rate, 0.001)
    printed = 0
    rows_sent = 0
    publish_count = 0

    for _, row in df.iterrows():
        if limit_rows_per_stream is not None and rows_sent >= limit_rows_per_stream:
            break

        node_id = str(row["node_id"])
        base_payload = {k: v for k, v in row.dropna().to_dict().items()}

        raw_val = row.get(ma_signal)
        try:
            raw_val = float(raw_val)
        except Exception:
            raw_val = 0.0

        for sid in range(1, streams + 1):
            stream_id_str = _stream_label(sid)
            topic = _topic(topic_prefix, stream_id_str, node_id)

            payload = dict(base_payload)
            payload["stream_id"] = stream_id_str

            adjusted_value = raw_val + adjuster_by_stream[sid].adjust()
            payload["Hz_adjusted"] = adjusted_value

            if use_ma:
                ma = _ensure_ma(ma_by_key, sid, node_id, ma_window, std_ddof)
                avg = ma.update(adjusted_value)
                std = ma.std()
                payload[ma_field] = float(avg)
                payload[std_field] = float(std)

                ma60_val = ma.average(60)
                ma120_val = ma.average(120)
                ma180_val = ma.average(180)
                ma240_val = ma.average(240)

                features = [
                    adjusted_value,
                    ma60_val,
                    ma120_val,
                    ma180_val,
                    ma240_val,
                    ma.std(60),
                    ma.std(120),
                ]
                payload["is_anom_rf"] = int(models["rf"].classify(features))
                payload["is_anom_lstm"] = int(models["lstm"].classify(features))
                payload["is_anom_svm"] = int(models["svm"].classify(features))
                payload["is_anom_xgb"] = int(models["xgb"].classify(features))
            else:
                payload.setdefault("is_anom_rf", 0)
                payload.setdefault("is_anom_lstm", 0)
                payload.setdefault("is_anom_svm", 0)
                payload.setdefault("is_anom_xgb", 0)

            body = json.dumps(payload, default=str)

            if printed < 6:
                preview = body if len(body) <= 240 else body[:240] + "...}"
                print(f"[replay] topic={topic} payload={preview}")
                printed += 1

            info = client.publish(topic, payload=body, qos=0, retain=False)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(f"[mqtt] publish failed rc={info.rc} topic={topic}")

            publish_count += 1

            if publish_count % 200 == 0:
                info.wait_for_publish(timeout=2.0)

            time.sleep(sleep)

        rows_sent += 1

        if rows_sent % 1000 == 0:
            print(f"[replay] rows_sent={rows_sent} (per stream)")

    return rows_sent


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"[replay] CSV not found: {csv_path.resolve()}")

    streams = max(1, int(args.streams))
    use_ma = args.ma_window is not None and args.ma_window > 0
    ma_field = args.ma_field or (f"moving_avg_{args.ma_signal}" if use_ma else "moving_avg")
    std_field = args.std_field or (f"moving_std_{args.ma_signal}" if use_ma else "moving_std")

    print(
        f"[replay] starting\n"
        f"  csv          = {csv_path}\n"
        f"  host:port    = {args.host}:{args.port}\n"
        f"  topic_prefix = {args.topic_prefix}\n"
        f"  streams      = {streams}\n"
        f"  rate         = {args.rate} msg/s (per stream)\n"
        f"  limit        = {args.limit} (rows per stream)\n"
        f"  chunksize    = {args.chunksize}\n"
        f"  loop         = {args.loop}\n"
        f"  moving_stats = {'ON' if use_ma else 'OFF'}"
        f"{'' if not use_ma else f' (signal={args.ma_signal}, window={args.ma_window}, avg_field={ma_field}, std_field={std_field})'}\n"
    )

    models = {
        "rf": RF_Classifier(),
        "lstm": LSTM_Classifier(),
        "svm": SVM_Classifier(),
        "xgb": XGB_Classifier(),
    }

    adjuster_by_stream: Dict[int, object] = {}
    for sid in range(1, streams + 1):
        adjuster_by_stream[sid] = data_adjust(magnitude=0.05, method="gaussian", period=300 + sid * 25)

    ma_by_key: Dict[Tuple[int, str], Node_MA] = {}

    client = _connect_mqtt(args.host, args.port, client_id="replayer", timeout_s=10)

    try:
        pass_idx = 0
        while True:
            pass_idx += 1
            print(f"[replay] pass #{pass_idx}")

            for chunk in pd.read_csv(csv_path, chunksize=args.chunksize):
                _validate_columns(chunk, need_ma=use_ma, ma_signal=args.ma_signal)

                _publish_rows(
                    df=chunk,
                    client=client,
                    topic_prefix=args.topic_prefix,
                    streams=streams,
                    rate=args.rate,
                    limit_rows_per_stream=args.limit,
                    use_ma=use_ma,
                    ma_by_key=ma_by_key,
                    ma_window=args.ma_window if use_ma else 0,
                    std_ddof=args.std_ddof,
                    ma_signal=args.ma_signal,
                    ma_field=ma_field,
                    std_field=std_field,
                    adjuster_by_stream=adjuster_by_stream,
                    models=models,
                )

                if args.limit is not None:
                    break

            if not args.loop:
                break
            if args.limit is not None:
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[replay] interrupted by user")
    finally:
        client.loop_stop()
        client.disconnect()
        print("[replay] done.")


if __name__ == "__main__":
    main()
