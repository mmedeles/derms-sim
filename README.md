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
