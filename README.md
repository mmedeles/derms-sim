# DERMS Lightweight Simulation (capstone)

A minimal Distributed Energy Resource Management System (DERMS) simulation used
for senior capstone work. The prototype simulates multiple DER nodes (solar,
wind, battery, EV), writes CSV telemetry, and includes a live Dash visualization.
This repository is staged to add MQTT-based communications, FDIA attacker
injection, anomaly detection, and evaluation next.

## Project goals / 5-stage pipeline
1. **Simulator** — Python generators produce DER telemetry (CSV + MQTT).  
2. **Communication** — MQTT (Mosquitto) broker to route telemetry between nodes.  
3. **FDIA Attacker** — configurable attack injection (bias, drift, spoof, replay).  
4. **Edge Detection** — sliding-window anomaly detectors (IsolationForest, LOF).  
5. **Evaluation** — precision/recall, false alarm, detection latency metrics.

## Repo layout
??

## TODO
1. Build Anomaly Detection script
2. Internal Manipulated Data
3. ML Model(s)
4. Find a way to display anomaly detection

# DERMS Simulation Setup and execution 

This document describes how to fully run and demonstrate the DERMS Resilience Testbed simulation

---

## ONE-TIME SETUP (do this once)

### 1) Open PowerShell at the project root and activate venv

### 2) Install Python dependencies
```powershell
pip install -r requirements.txt
```

### 3) Create Mosquitto config (allows local connections)
```powershell
mkdir -Force mosquitto\config
@"
listener 1883
allow_anonymous true
"@ | Out-File -Encoding ascii mosquitto\config\mosquitto.conf
```

### 4) Normalize the advisor dataset (CSV → canonical format)
```powershell
python -m data.load_dataset --input "path/to/dataset" --config configs/dataset_map.yml --output data/normalized.csv
```

You should see:
```
[normalize] Wrote normalized CSV → data\normalized.csv
```

> **Reminder:**  
> Make sure `viz/app.py` has the **single on_message handler** fix and subscribes to:
> - `derms/+/telemetry`  
> - `derms/+/anomaly`

---

## The Runbook

### 1) Start the MQTT broker (Docker)

If a previous container exists, remove it first:
```powershell
docker rm -f mosq
```

Now start Mosquitto with your config mounted:
```powershell
docker run -it --name mosq `
  -p 1883:1883 `
  -v "path\to\derms-sim\mosquitto\config:/mosquitto/config" `
  eclipse-mosquitto
```

Leave this window open.

You should see:
```
Config loaded from /mosquitto/config/mosquitto.conf
Opening ipv4 listen socket on port 1883
```
 *There should be no “local only mode” message.*

---

### 2) (Optional) Live topic monitor (proves broker is passing messages)
Open a **new PowerShell window**:
```powershell
docker exec -it mosq sh -lc "mosquitto_sub -t '#' -v"
```

Leave it running - it will print messages whenever publishers send data.

---

### 3) Start the Dash app (in another PowerShell window)
```powershell
python -m viz.app
```

Then open in your browser:
 [http://127.0.0.1:8050](http://127.0.0.1:8050)

You should see:
```
MQTT host: localhost • Client: OK • Nodes observed: - waiting -
```

---

### 4) Start the dataset replayer (publisher)

Continuous? ==> dashboard always has fresh data:
```powershell
python -m data.replay_mqtt --csv data/normalized.csv --rate 20 --host localhost --loop --ma-window 240
```

Within a few seconds, the **Node dropdown** should populate (e.g., `inverter_1`),  
and the **Power / Voltage / Frequency** graphs will start updating live.

short burst instead?
```powershell
python -m data.replay_mqtt --csv data/normalized.csv --rate 200 --limit 2000 --host localhost
```

While running, your `mosquitto_sub` window should show lines like:
```
derms/inverter_1/telemetry { ... }
```

*review script documentation for command(s) options and values*

---

### 5) (Optional) Start the anomaly detector...
#### STILL IN THE WORKS
```powershell
python -m detector.zscore --host localhost --window 200 --threshold 3.0
```

- The **Dash “Anomaly” banner** will flip when z-scores exceed the threshold.  
- The subscriber window will show:
  ```
  derms/inverter_1/anomaly { ... }
  ```

---

## WHAT YOU SHOULD SEE (sanity checklist)

| Component | Expected Behavior |
|------------|------------------|
| **Broker window** | “Opening listen socket on port 1883” |
| **Subscriber window** | Topic lines like `derms/inverter_1/telemetry { ... }` |
| **Replayer window** | Startup banner and `[replay] sent=1000` progress prints |
| **Dash window** | Node listed (e.g., inverter_1), charts animate live every second |

---

## COMMON QUICK FIXES

| Issue | Solution |
|--------|-----------|
| **Port conflict / old container** | Run `docker rm -f mosq` then restart Step 1 |
| **No messages in subscriber** | Ensure replayer prints topics starting with `derms/` (default) |
| **Dash not updating** | Verify the single `on_message` handler is used in `viz/app.py`; restart Dash and replayer |
| **Windows path errors** | Use absolute path in `-v` mount (see exact path in Step 1) |

---

## CLEANUP

Stop the broker in its window with **Ctrl+C**, then remove the container:
```powershell
docker rm -f mosq
```

Deactivate the virtual environment:
```powershell
deactivate
```

---

