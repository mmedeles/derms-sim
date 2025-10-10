"""
Replay normalized CSV rows to MQTT topics at a fixed rate.

Input CSV schema (normalized):
  timestamp,node_id,type,power_kw,voltage,frequency_hz

Publishes (JSON):
  topic: {topic_prefix}/{node_id}/telemetry
  payload: row values as dict (non-null)

Examples:
  python -m data.replay_mqtt --csv data/normalized.csv --rate 10 --host localhost
  python -m data.replay_mqtt --csv data/normalized.csv --rate 200 --limit 2000 --host localhost
  python -m data.replay_mqtt --csv data/normalized.csv --rate 20 --host localhost --loop
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from comm.mqtt_client import MQTTClient


def parse_args():
    ap = argparse.ArgumentParser(description="Replay normalized CSV to MQTT.")
    ap.add_argument("--csv", default="data/normalized.csv", help="Path to normalized CSV")
    ap.add_argument("--host", default="localhost", help="MQTT broker host")
    ap.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    ap.add_argument("--topic-prefix", dest="topic_prefix", default="derms", help="Topic prefix (default: derms)")
    ap.add_argument("--rate", type=float, default=10.0, help="Messages per second")
    ap.add_argument("--limit", type=int, default=None, help="Optional max rows to send (then exit)")
    ap.add_argument("--chunksize", type=int, default=10_000, help="CSV read chunk size")
    ap.add_argument("--loop", action="store_true", help="Loop over the CSV forever")
    return ap.parse_args()


REQUIRED_COLS = ["timestamp", "node_id", "power_kw"]


def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[replay] CSV is missing required columns: {missing}. "
            "Expected at least: timestamp,node_id,power_kw (plus optional voltage,frequency_hz)."
        )


def publish_dataframe(df: pd.DataFrame, mqc: MQTTClient, topic_prefix: str, rate: float, limit: int | None):
    sleep = 1.0 / max(rate, 0.001)
    sent = 0
    printed = 0

    validate_columns(df)

    for _, row in df.iterrows():
        node = str(row["node_id"])
        topic = f"{topic_prefix}/{node}/telemetry"

        # Build clean payload
        payload = {k: v for k, v in row.dropna().to_dict().items()}

        # Debug: show first few
        if printed < 5:
            print(f"[replay] host={mqc.client._host} topic={topic} payload={json.dumps(payload)[:200]}")
            printed += 1

        mqc.publish(topic, payload)
        sent += 1

        if sent % 1000 == 0:
            print(f"[replay] sent={sent}")

        if limit and sent >= limit:
            print(f"[replay] sent {sent} messages; exiting")
            return sent

        time.sleep(sleep)

    return sent


def one_pass(csv_path: Path, chunksize: int, mqc: MQTTClient, topic_prefix: str, rate: float, limit: int | None):
    """Stream the CSV once, chunk by chunk."""
    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        validate_columns(chunk)
        total += publish_dataframe(chunk, mqc, topic_prefix, rate, limit=None if limit is None else max(0, limit - total))
        if limit is not None and total >= limit:
            break
    return total


def main():
    args = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"[replay] CSV not found: {csv_path.resolve()}")

    print(
        f"[replay] starting\n"
        f"  csv          = {csv_path}\n"
        f"  host:port    = {args.host}:{args.port}\n"
        f"  topic_prefix = {args.topic_prefix}\n"
        f"  rate         = {args.rate} msg/s\n"
        f"  limit        = {args.limit}\n"
        f"  chunksize    = {args.chunksize}\n"
        f"  loop         = {args.loop}\n"
    )

    mqc = MQTTClient(client_id="replayer", host=args.host, port=args.port)

    try:
        if args.loop:
            # Loop forever over the CSV
            #TODO: Find A Way to indicate the end of the CSV when loop cycles through the data in full
            #end of csv/data must be seen on the dash, broker, and replayer

            loop_count = 0
            while True:
                loop_count += 1
                print(f"[replay] pass #{loop_count}")
                sent = one_pass(csv_path, args.chunksize, mqc, args.topic_prefix, args.rate, args.limit)
                if args.limit is not None:
                    # With a limit, one pass will always exit early → stop
                    break
                # Small pause between loops (so the UI clearly shows wrap-around)
                time.sleep(1.0)
        else:
            # Single pass
            one_pass(csv_path, args.chunksize, mqc, args.topic_prefix, args.rate, args.limit)

    except KeyboardInterrupt:
        print("\n[replay] interrupted by user")
    except Exception as e:
        print(f"[replay] ERROR: {e}")
        raise
    finally:
        print("[replay] done.")


if __name__ == "__main__":
    main()
