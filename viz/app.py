"""
Dash visualization that CONSUMES live telemetry over MQTT.

- Telemetry: derms/+/telemetry
- Anomalies: derms/+/anomaly (optional)

Shows:
  * Power (kW), Voltage (V), Frequency (Hz)
  * Frequency overlays: Moving Avg (dotted) and ±k*STD shaded band (if present)

Env:
  MQTT_HOST             (default: localhost)
  HISTORY_POINTS        (default: 1200)
  PLOT_POINTS           (default: 300)
  MA_FIELD              (default: moving_avg_frequency_hz)
  STD_FIELD             (default: moving_std_frequency_hz)
  STD_BAND_FACTOR       (default: 1.0)  # width of shaded band = MA ± k*STD
"""

from __future__ import annotations
import os, json, threading
from collections import deque, defaultdict
from datetime import datetime
from typing import List, Tuple

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
try:
    from comm.mqtt_client import MQTTClient
except Exception:
    MQTTClient = None

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
HISTORY_POINTS = int(os.getenv("HISTORY_POINTS", "1200"))
PLOT_POINTS = int(os.getenv("PLOT_POINTS", "300"))

MA_FIELD = os.getenv("MA_FIELD", "moving_avg_frequency_hz")
STD_FIELD = os.getenv("STD_FIELD", "moving_std_frequency_hz")
STD_BAND_FACTOR = float(os.getenv("STD_BAND_FACTOR", "1.0"))

BUFFER = defaultdict(lambda: deque(maxlen=HISTORY_POINTS))     # node_id -> deque of telemetry dicts
ANOMALIES = defaultdict(lambda: deque(maxlen=HISTORY_POINTS))  # node_id -> deque of anomaly dicts
KNOWN_NODES = set()

# ------------------------------------------------------------------------------
# MQTT subscriber
# ------------------------------------------------------------------------------
def _start_mqtt():
    if MQTTClient is None:
        print("[viz] MQTTClient not available.")
        return

    mqc = MQTTClient(client_id="viz_ui", host=MQTT_HOST)

    def on_any(_c, _u, msg):
        topic = msg.topic or ""
        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return

        if topic.endswith("/telemetry"):
            nid = str(data.get("node_id", "unknown"))
            if data.get("timestamp"):
                BUFFER[nid].append(data)
                KNOWN_NODES.add(nid)
        elif topic.endswith("/anomaly"):
            nid = str(data.get("node_id", "unknown"))
            ANOMALIES[nid].append(data)

    mqc.client.on_message = on_any
    mqc.client.subscribe("derms/+/telemetry")
    mqc.client.subscribe("derms/+/anomaly")
    print(f"[viz] Subscribed to MQTT at {MQTT_HOST} (derms/+/telemetry, derms/+/anomaly)")

threading.Thread(target=_start_mqtt, daemon=True).start()

# ------------------------------------------------------------------------------
# Dash app
# ------------------------------------------------------------------------------
app = dash.Dash(__name__)
server = app.server
app.title = "DERMS Live Dashboard"

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, sans-serif", "padding": "16px", "maxWidth": "1200px", "margin": "0 auto"},
    children=[
        html.H2("DERMS Live Dashboard (MQTT)"),
        html.P(
            id="mqtt-status",
            style={"padding":"10px","borderRadius":"8px","backgroundColor":"#f3f4f6","border":"1px solid #e5e7eb","fontSize":"14px","marginBottom":"12px"},
        ),
        html.Div(
            style={"display":"flex","gap":"12px","alignItems":"center","flexWrap":"wrap"},
            children=[
                html.Label("Node:", style={"fontWeight": 600}),
                dcc.Dropdown(id="node-select", options=[], value=None, placeholder="Waiting for data…", style={"minWidth":"240px"}, clearable=False),
                html.Div(id="anomaly-banner", style={"fontWeight": 600}),
            ],
        ),
        html.Div(style={"height":"12px"}),
        dcc.Graph(id="power-graph", config={"displayModeBar": False}),
        dcc.Graph(id="voltage-graph", config={"displayModeBar": False}),
        dcc.Graph(id="frequency-graph", config={"displayModeBar": False}),
        dcc.Interval(id="tick", interval=1000, n_intervals=0),
    ],
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _parse_ts(ts: str):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def _get_series_for_node(node_id: str, max_points: int = PLOT_POINTS):
    if not node_id or node_id not in BUFFER or len(BUFFER[node_id]) == 0:
        return [], [], [], [], [], []

    items = list(BUFFER[node_id])[-max_points:]
    parsed = []
    for d in items:
        t = _parse_ts(d.get("timestamp",""))
        if t is not None:
            parsed.append((t, d))

    if not parsed:
        return [], [], [], [], [], []

    times = [t for (t, _) in parsed]
    p_kw  = [float(x.get("power_kw", 0.0)) for _, x in parsed]
    v     = [float(x.get("voltage", 0.0))   for _, x in parsed]
    f_hz  = [float(x.get("frequency_hz", 0.0)) for _, x in parsed]

    ma_vals, std_vals, adjusted_vals, is_anoms = [], [], [], []
    for _, x in parsed:
        # moving average
        mv = x.get(MA_FIELD)
        adjusted = x.get("Hz_adjusted")
        try:
            mv = float(mv) if mv is not None else None
            adjusted = float(adjusted) if adjusted is not None else None
        except Exception:
            mv = None
            adjusted = None
        ma_vals.append(mv)
        is_anoms.append(x.get("is_anom"))
        adjusted_vals.append(adjusted)
        # moving std
        sv = x.get(STD_FIELD)
        try:
            sv = float(sv) if sv is not None else None
        except Exception:
            sv = None
        std_vals.append(sv)

    return times, p_kw, v, f_hz, ma_vals, std_vals, adjusted_vals, is_anoms

def _recent_anomaly_label(node_id: str):
    if node_id in ANOMALIES and len(ANOMALIES[node_id]) > 0:
        recent = list(ANOMALIES[node_id])[-50:]
        if any(bool(a.get("anomaly")) for a in recent):
            return ("Anomaly detected in recent samples", "#fee2e2", "#991b1b")
    return ("No recent anomalies", "#ecfdf5", "#065f46")

def _line_figure(title: str, x, y, y_title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=title))
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=10, t=70, b=40),
        height=300,
        xaxis=dict(tickformat="%H:%M:%S", tickmode="auto", nticks=8, showgrid=True),
        yaxis_title=y_title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        uirevision="keep"
    )
    return fig

# ------------------------------------------------------------------------------
# Callback
# ------------------------------------------------------------------------------
@app.callback(
    Output("mqtt-status","children"),
    Output("node-select","options"),
    Output("node-select","value"),
    Output("anomaly-banner","children"),
    Output("anomaly-banner","style"),
    Output("power-graph","figure"),
    Output("voltage-graph","figure"),
    Output("frequency-graph","figure"),
    Input("tick","n_intervals"),
    State("node-select","value"),
)
def update_ui(_n, selected_node):
    mqtt_ok = MQTTClient is not None
    have_data = len(KNOWN_NODES) > 0
    status = f"MQTT host: {MQTT_HOST}  •  Client: {'OK' if mqtt_ok else 'Unavailable'}  •  Nodes observed: {', '.join(sorted(KNOWN_NODES)) if have_data else '— waiting —'}"

    options = [{"label": nid, "value": nid} for nid in sorted(KNOWN_NODES)]
    if selected_node is None and options:
        selected_node = options[0]["value"]

    if selected_node and selected_node in BUFFER:
        times, p_kw, v, f_hz, ma_vals, std_vals, adjusted_vals, is_anoms = _get_series_for_node(selected_node, max_points=PLOT_POINTS)
        power_fig = _line_figure("Power (kW)", times, p_kw, "kW")
        volt_fig  = _line_figure("Voltage (V)", times, v, "V")
        freq_fig  = _line_figure("Frequency (Hz)", times, f_hz, "Hz")

        # Add Moving Avg (dotted) if present
        if any(val is not None for val in ma_vals):
            freq_fig.add_trace(go.Scatter(x=times, y=ma_vals, mode="lines", name="Moving Avg", line=dict(dash="dot")))
        if any(val is not None for val in adjusted_vals):
            symbols = ['circle' if not anom else "x" for anom in is_anoms]
            sizes = [20 if anom else 6 for anom in is_anoms]
            colors = ['red' if anom else 'blue' for anom in is_anoms]
            freq_fig.add_trace(go.Scatter(x=times, y=adjusted_vals, 
                                          mode="markers+lines", 
                                          name="Adjusted Values", 
                                          marker=dict(symbol=symbols,
                                                      size=sizes,
                                                      color=colors,
                                                      line=dict(width=1)) ))
        #if any(val is not None for val in is_anoms):
            #symbols = ['circle' if not anom else "x" for anom in is_anoms]
            #freq_fig.add_trace(go.Scatter(x=times, y=adjusted_vals, mode="markers", name="Anom Classification", line=dict(dash="dot")))

        # Add ±k*STD band around MA if both series present
        if STD_BAND_FACTOR > 0 and any(s is not None for s in std_vals) and any(m is not None for m in ma_vals):
            upper = []
            lower = []
            for m, s in zip(ma_vals, std_vals):
                if m is None or s is None:
                    upper.append(None); lower.append(None)
                else:
                    upper.append(m + STD_BAND_FACTOR * s)
                    lower.append(m - STD_BAND_FACTOR * s)
            # shaded band
            freq_fig.add_trace(go.Scatter(
                x=times, y=upper, mode="lines", name=f"+{STD_BAND_FACTOR}·STD",
                line=dict(width=0), showlegend=False
            ))
            freq_fig.add_trace(go.Scatter(
                x=times, y=lower, mode="lines", name=f"-{STD_BAND_FACTOR}·STD",
                line=dict(width=0), fill="tonexty", fillcolor="rgba(59,130,246,0.15)", showlegend=False
            ))

        msg, bg, fg = _recent_anomaly_label(selected_node)
        banner_style = {"padding":"6px 10px","borderRadius":"8px","backgroundColor":bg,"color":fg}
        return status, options, selected_node, msg, banner_style, power_fig, volt_fig, freq_fig

    placeholder = _line_figure("", [], [], "")
    placeholder.update_layout(showlegend=False)
    banner_style = {"padding":"6px 10px","borderRadius":"8px","backgroundColor":"#fef3c7","color":"#92400e"}
    return (status, options, selected_node,
            "Waiting for data… start the replayer to see live updates.",
            banner_style, placeholder, placeholder, placeholder)

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True, use_reloader=False)
