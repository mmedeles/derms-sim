"""
Simulator runner / CLI script.

Purpose
-------
Drive a multi-node simulation, write CSV output, and (optionally) publish JSON
telemetry messages to an MQTT broker. This file is intended to be the
single-run entrypoint for quick demos and for integration into the comms
pipeline during later stages.

Key features
------------
- Instantiates a configurable set of DERNode objects (from sim.nodes).
- Advances the simulation for a configurable number of points or indefinitely.
- Writes telemetry rows to a CSV file via sim.log.Log.
- Optionally publishes each sample to MQTT topics (requires paho-mqtt and an
  MQTT client wrapper; publishing is off by default to preserve your current
  CSV-only workflow).

CLI Usage (examples)
--------------------
# Run 100 samples at 1Hz and write CSV (default behavior)
python sim/run_sim.py --points 100 --rate 1.0 --csv sim_output.csv

# Publish to local MQTT broker (host=localhost) while also writing CSV
python sim/run_sim.py --points 200 --rate 2.0 --publish --mqtt-host localhost --csv sim_output.csv

# Run indefinitely with publishing enabled (stop with Ctrl+C)
python sim/run_sim.py --rate 1.0 --publish

Public API
----------
- main(args: argparse.Namespace) -> None
    Primary function invoked when run as a script. Parses args, constructs nodes,
    and runs the publish/write loop.

Important CLI arguments
-----------------------
--points INT         : number of samples to produce (if omitted, run forever)
--rate FLOAT         : samples per second (default 1.0)
--csv PATH           : path to write CSV output (default 'sim_output.csv')

Outputs / Side Effects
----------------------
- Writes CSV file to disk.
- Optionally publishes telemetry JSON to MQTT topics:
  topic pattern: derms/{node_id}/telemetry

Dependencies
------------
- sim.nodes, sim.log
- optionally comm.mqtt_client (if --publish used)
- python packages: pandas, numpy (already used by nodes/log)

Testing
-------
- Unit tests should mock MQTT client when `--publish` is used.
- For CSV-only runs, run with small point counts and assert CSV file has the
  expected number of rows and expected header columns.

Development notes
-----------------
- The current code writes full CSV each run; for extremely long runs consider
  chunked/append mode or rotating log files.
- When integrating with live visualization or attacker/detector modules, use
  configuration files to describe node lists instead of hard-coded lists.
"""

import argparse
from datetime import datetime, timedelta
from sim.nodes import DERNode, SolarNode
from sim.log import Log

def main(a):
    nodes = [
        SolarNode("solar1"),
        DERNode("wind1",  "wind",  base_p_kw=4.0),
        DERNode("bat1",   "battery", base_p_kw=2.5),
        DERNode("ev1",    "ev",    base_p_kw=7.0),
    ]
    log = Log(nodes, scenario="normal")
    now = datetime.utcnow()

    # simulate N points at step seconds
    for i in range(0,a.points,a.step):
        for n in nodes:
            n.step(dt=a.step)
        ts = now + timedelta(seconds=i*a.step)
        # store ISO timestamp
        log.add_row(ts.isoformat())

    out = a.out or "sim_output.csv"
    log.to_csv(out)
    print(f"Saved {len(log.rows)} rows to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=int, default=500000, help="number of samples")
    p.add_argument("--step", type=int, default=1000, help="seconds between samples")
    p.add_argument("--out", type=str, default="sim_output.csv")
    args = p.parse_args()
    main(args)
