"""
Log helper for the simulator.

Purpose
-------
Collects telemetry rows produced by the simulation and writes structured CSV
output for offline analysis, reproducible runs, and as a ground-truth artifact
for evaluation. Uses pandas to write CSV.

Why this exists
----------------
- Keeps a single place for CSV header and write formatting.
- Simplifies switching between CSV output and MQTT streaming.
- Produces reproducible artifacts for demo and evaluation.

Public API
----------
class Log:
    - __init__(self, headers: List[str]) -> None
        Create a new Log object with column headers (list of string names).
    - append(self, row: Dict[str, Any]) -> None
        Append a dictionary representing one telemetry row. Keys should map to
        headers or will be added to CSV as additional columns.
    - to_csv(self, path: str) -> None
        Write the accumulated rows to disk as CSV (uses pandas.DataFrame).

Example
-------
from sim.log import Log
log = Log(headers=["timestamp", "node_id", "power_kw", "voltage"])
log.append({"timestamp": "2025-10-01T12:00:00Z", "node_id": "n1", "power_kw": 3.91, "voltage": 120.1})
log.to_csv("sim_output.csv")

Expected Inputs/Outputs
-----------------------
- Input: dictionaries from the simulator nodes.
- Output: CSV file at provided path, with UTF-8 encoding and comma delimiter.

Dependencies
------------
- pandas

Testing
-------
- Unit tests should check that appended rows appear in the saved CSV and that
  DataFrame column order matches headers passed at init when possible.

TODO / Extensions
-----------------
- Add append-to-file streaming (append mode) for long duration simulations.
- Add a `to_parquet` or `to_jsonl` option for more efficient storage.
- Add timestamps and run metadata (seed, profile config) to the CSV header.
"""

import pandas as pd
import matplotlib.pyplot as plt

class Log:
    def __init__(self, nodes, scenario="normal"):
        self.rows = []
        self.nodes = nodes
        self.scenario = scenario
        # headers: time, then for each node: P, V, F
        self.columns = ["time"]
        for n in nodes:
            self.columns += [f"{n.name}_P", f"{n.name}_V", f"{n.name}_F"]
        self.columns.append("scenario")

    def add_row(self, time):
        row = [time]
        for n in self.nodes:
            row += [n.base_p_kw, n.v_pu, n.freq_hz]
        row.append(self.scenario)
        self.rows.append(row)

    def to_df(self):
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_csv(self, path):
        self.to_df().to_csv(path, index=False)
    
    def plot_time_series(self):
        data=self.to_df()
        plt.figure(figsize=(16,8))
        #Target variable as "target"
        for col in data.drop(columns="target",errors="ignore").columns:
            plt.plot(data.index, data[col], label=col)
        
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Time Series Values")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
