"""
Replay normalized CSV rows to MQTT topics at a fixed rate,
with enriching each row with moving average & moving std.

Expected input CSV columns (normalized):
  timestamp,node_id,type,power_kw,voltage,frequency_hz

Publishes JSON to:
  {topic_prefix}/{node_id}/telemetry

Examples:
  # continuous replay + MA & STD on frequency_hz
  python -m data.replay_mqtt --csv data/normalized.csv --host localhost --rate 20 --loop \
      --ma-window 240 --ma-signal frequency_hz \
      --ma-field moving_avg_frequency_hz --std-field moving_std_frequency_hz
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from comm.mqtt_client import MQTTClient
from .moving_avg import Node_MA

# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Replay normalized CSV to MQTT.")
    ap.add_argument("--csv", default="data/normalized.csv", help="Path to normalized CSV")
    ap.add_argument("--host", default="localhost", help="MQTT broker host")
    ap.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    ap.add_argument("--topic-prefix", dest="topic_prefix", default="derms", help="Topic prefix")
    ap.add_argument("--rate", type=float, default=10.0, help="Messages per second")
    ap.add_argument("--limit", type=int, default=None, help="Max rows to send then exit")
    ap.add_argument("--chunksize", type=int, default=10_000, help="CSV read chunk size")
    ap.add_argument("--loop", action="store_true", help="Loop over the CSV forever")

    # Moving statistics
    ap.add_argument("--ma-window", type=int, default=None, help="Enable moving stats with this window size")
    ap.add_argument("--ma-signal", default="frequency_hz", help="Column to smooth (e.g., frequency_hz or power_kw)")
    ap.add_argument("--ma-field", default=None, help="Output field for moving average (default: moving_avg_<signal>)")
    ap.add_argument("--std-field", default=None, help="Output field for moving std (default: moving_std_<signal>)")
    ap.add_argument("--std-ddof", type=int, default=1, help="ddof for sample std (default 1)")
    return ap.parse_args()

REQUIRED_COLS = ["timestamp", "node_id", "power_kw"]

def _validate_columns(df: pd.DataFrame, need_ma: bool, ma_signal: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[replay] CSV missing required columns: {missing}")
    if need_ma and ma_signal not in df.columns:
        raise ValueError(f"[replay] --ma-signal '{ma_signal}' not found in CSV columns.")

# ---------------------------------------------------------------------------

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
):
    sleep = 1.0 / max(rate, 0.001)
    sent = 0
    printed = 0

    for _, row in df.iterrows():
        node = str(row["node_id"])
        topic = f"{topic_prefix}/{node}/telemetry"

        payload = {k: v for k, v in row.dropna().to_dict().items()}

        # add moving stats if enabled
        if ma is not None:
            avg = ma.update(row.get(ma_signal))
            std = ma.std()
            payload[ma_field] = float(avg)
            payload[std_field] = float(std)

        if printed < 5:
            preview = json.dumps(payload, default=str)
            if len(preview) > 240:
                preview = preview[:240] + "...}"
            print(f"[replay] host={mqc.client._host} topic={topic} payload={preview}")
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
):
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
        )
        if limit is not None and total >= limit:
            break
    return total

# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"[replay] CSV not found: {csv_path.resolve()}")

    use_ma = args.ma_window is not None and args.ma_window > 0
    ma = Node_MA(n=args.ma_window, ddof=args.std_ddof) if use_ma else None
    ma_field = args.ma_field or (f"moving_avg_{args.ma_signal}" if use_ma else "")
    std_field = args.std_field or (f"moving_std_{args.ma_signal}" if use_ma else "")

    print(
        f"[replay] starting\n"
        f"  csv          = {csv_path}\n"
        f"  host:port    = {args.host}:{args.port}\n"
        f"  topic_prefix = {args.topic_prefix}\n"
        f"  rate         = {args.rate} msg/s\n"
        f"  limit        = {args.limit}\n"
        f"  chunksize    = {args.chunksize}\n"
        f"  loop         = {args.loop}\n"
        f"  moving_stats = {'ON' if use_ma else 'OFF'}"
        f"{'' if not use_ma else f' (signal={args.ma_signal}, window={args.ma_window}, avg_field={ma_field}, std_field={std_field})'}\n"
    )

    mqc = MQTTClient(client_id="replayer", host=args.host, port=args.port)

    try:
        if args.loop:
            pass_idx = 0
            while True:
                pass_idx += 1
                print(f"[replay] pass #{pass_idx}")
                _one_pass(
                    csv_path, args.chunksize, mqc, args.topic_prefix, args.rate,
                    args.limit, ma, args.ma_signal, ma_field, std_field
                )
                if args.limit is not None:
                    break
                time.sleep(0.5)
        else:
            _one_pass(
                csv_path, args.chunksize, mqc, args.topic_prefix, args.rate,
                args.limit, ma, args.ma_signal, ma_field, std_field
            )
    except KeyboardInterrupt:
        print("\n[replay] interrupted by user")
    except Exception as e:
        print(f"[replay] ERROR: {e}")
        raise
    finally:
        print("[replay] done.")

if __name__ == "__main__":
    main()
