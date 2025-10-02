import argparse
from datetime import datetime, timedelta
from sim.nodes import DERNode
from sim.log import Log

def main(a):
    nodes = [
        DERNode("solar1", "solar", base_p_kw=5.0),
        DERNode("wind1",  "wind",  base_p_kw=4.0),
        DERNode("bat1",   "battery", base_p_kw=2.5),
        DERNode("ev1",    "ev",    base_p_kw=7.0),
    ]
    log = Log(nodes, scenario="normal")
    now = datetime.utcnow()

    # simulate N points at step seconds
    for i in range(a.points):
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
    p.add_argument("--points", type=int, default=600, help="number of samples")
    p.add_argument("--step", type=int, default=1, help="seconds between samples")
    p.add_argument("--out", type=str, default="sim_output.csv")
    args = p.parse_args()
    main(args)
