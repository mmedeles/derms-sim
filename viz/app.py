"""
Dash visualization that CONSUMES live telemetry over MQTT.

Subscribes to:
  - derms/+/telemetry   (timestamp,node_id,type,power_kw,voltage,frequency_hz[, moving_avg_*])
  - derms/+/anomaly     (optional anomaly messages)

Shows:
  - Node selector (auto-discovers node_ids)
  - Live graphs: Power (kW), Voltage (V), Frequency (Hz)
  - Frequency plot overlays a dotted Moving Average (if present in messages)
  - Simple anomaly banner

Run:
  python -m viz.app
  # open http://127.0.0.1:8050
"""

from __future__ import annotations
import os
import json
import threading
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
    MQTTClient = None  # UI will still load; status will say Unavailable.

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")

# how many points to keep per node in memory and to plot
HISTORY_POINTS = int(os.getenv("HISTORY_POINTS", "1200"))
PLOT_POINTS = int(os.getenv("PLOT_POINTS", "300"))  # show last N on each refresh

# The field name used for moving average overlay (published by replayer)
# default matches: --ma-field moving_avg_frequency_hz
MA_FIELD = os.getenv("MA_FIELD", "moving_avg_frequency_hz")

# Global buffers (in-memory, process-local)
BUFFER = defaultdict(lambda: deque(maxlen=HISTORY_POINTS))     # node_id -> deque of telemetry dicts
ANOMALIES = defaultdict(lambda: deque(maxlen=HISTORY_POINTS))  # node_id -> deque of anomaly dicts
KNOWN_NODES = set()


# ------------------------------------------------------------------------------
# MQTT subscriber (single on_message handler)
# ------------------------------------------------------------------------------
def _start_mqtt():
    """Background MQTT subscriber thread that fills BUFFER and ANOMALIES."""
    if MQTTClient is None:
        print("[viz] MQTTClient not available; cannot subscribe to MQTT.")
        return

    mqc = MQTTClient(client_id="viz_ui", host=MQTT_HOST)

    def on_any(_client, _userdata, msg):
        topic = msg.topic or ""
        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return

        if topic.endswith("/telemetry"):
            node_id = str(data.get("node_id", "unknown"))
            if data.get("timestamp"):
                BUFFER[node_id].append(data)
                KNOWN_NODES.add(node_id)
        elif topic.endswith("/anomaly"):
            node_id = str(data.get("node_id", "unknown"))
            ANOMALIES[node_id].append(data)

    # Attach one handler, subscribe to both patterns
    mqc.client.on_message = on_any
    mqc.client.subscribe("derms/+/telemetry")
    mqc.client.subscribe("derms/+/anomaly")
    print(f"[viz] Subscribed to MQTT at {MQTT_HOST} (derms/+/telemetry, derms/+/anomaly)")


# Start MQTT listener in this (serving) process
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
            style={
                "padding": "10px",
                "borderRadius": "8px",
                "backgroundColor": "#f3f4f6",
                "border": "1px solid #e5e7eb",
                "fontSize": "14px",
                "marginBottom": "12px",
            },
        ),

        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
            children=[
                html.Label("Node:", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="node-select",
                    options=[],
                    value=None,
                    placeholder="Waiting for data…",
                    style={"minWidth": "240px"},
                    clearable=False,
                ),
                html.Div(id="anomaly-banner", style={"fontWeight": 600}),
            ],
        ),

        html.Div(style={"height": "12px"}),

        dcc.Graph(id="power-graph", figure=go.Figure(), config={"displayModeBar": False}),
        dcc.Graph(id="voltage-graph", figure=go.Figure(), config={"displayModeBar": False}),
        dcc.Graph(id="frequency-graph", figure=go.Figure(), config={"displayModeBar": False}),

        # UI refresh cadence (ms)
        dcc.Interval(id="tick", interval=1000, n_intervals=0),
    ],
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _parse_ts(ts: str):
    """Parse ISO8601 timestamps (Z or offset) → datetime; return None if invalid."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _get_series_for_node(node_id: str, max_points: int = PLOT_POINTS):
    """Extract recent time series for a node from BUFFER using real datetime.
    Returns: times, power_kW, voltage_V, frequency_Hz, moving_avg (or list of None)
    """
    if not node_id or node_id not in BUFFER or len(BUFFER[node_id]) == 0:
        return [], [], [], [], []

    items = list(BUFFER[node_id])[-max_points:]
    parsed = []
    for d in items:
        t = _parse_ts(d.get("timestamp", ""))
        if t is not None:
            parsed.append((t, d))

    if not parsed:
        return [], [], [], [], []

    times = [t for (t, _) in parsed]
    p_kw  = [float(x.get("power_kw", 0.0)) for _, x in parsed]
    v     = [float(x.get("voltage", 0.0))   for _, x in parsed]
    f_hz  = [float(x.get("frequency_hz", 0.0)) for _, x in parsed]

    # moving average
    ma_vals = []
    for _, x in parsed:
        val = x.get(MA_FIELD)
        try:
            ma_vals.append(float(val) if val is not None else None)
        except Exception:
            ma_vals.append(None)

    return times, p_kw, v, f_hz, ma_vals

def _recent_anomaly_label(node_id: str):
    """Return a small banner message if any anomalies recently flagged."""
    if node_id in ANOMALIES and len(ANOMALIES[node_id]) > 0:
        recent = list(ANOMALIES[node_id])[-50:]
        if any(bool(a.get("anomaly")) for a in recent):
            return ("Anomaly detected in recent samples", "#fee2e2", "#991b1b")  # red
    return ("No recent anomalies", "#ecfdf5", "#065f46")  # green


def _line_figure(title: str, x, y, y_title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=title))
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=10, t=40, b=40),
        height=300,
        xaxis=dict(
            tickformat="%H:%M:%S",   # pretty HH:MM:SS labels
            tickmode="auto",
            nticks=8,                # limit tick count
            showgrid=True,
        ),
        yaxis_title=y_title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-.25, xanchor="left", x=0),
    )
    return fig

# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------
@app.callback(
    Output("mqtt-status", "children"),
    Output("node-select", "options"),
    Output("node-select", "value"),
    Output("anomaly-banner", "children"),
    Output("anomaly-banner", "style"),
    Output("power-graph", "figure"),
    Output("voltage-graph", "figure"),
    Output("frequency-graph", "figure"),
    Input("tick", "n_intervals"),
    State("node-select", "value"),
)
def update_ui(_n, selected_node):
    # 1) MQTT status text
    mqtt_ok = MQTTClient is not None
    have_data = len(KNOWN_NODES) > 0
    status = (
        f"MQTT host: {MQTT_HOST}  •  Client: {'OK' if mqtt_ok else 'Unavailable'}  •  "
        f"Nodes observed: {', '.join(sorted(KNOWN_NODES)) if have_data else '— waiting —'}"
    )

    # 2) Node options & default selection
    options = [{"label": nid, "value": nid} for nid in sorted(KNOWN_NODES)]
    if selected_node is None and options:
        selected_node = options[0]["value"]

    # 3) Series for graphs
    if selected_node and selected_node in BUFFER:
        times, p_kw, v, f_hz, ma_vals = _get_series_for_node(selected_node, max_points=PLOT_POINTS)
        power_fig = _line_figure("Power (kW)", times, p_kw, "kW")
        volt_fig  = _line_figure("Voltage (V)", times, v, "V")
        freq_fig  = _line_figure("Frequency (Hz)", times, f_hz, "Hz")

        # Add moving-average overlay on Frequency (dotted line) if present
        if any(val is not None for val in ma_vals):
            freq_fig.add_trace(
                go.Scatter(
                    x=times,
                    y=ma_vals,
                    mode="lines",
                    name="Moving Avg",
                    line=dict(dash="dot")
                )
            )

        # 4) Anomaly banner
        msg, bg, fg = _recent_anomaly_label(selected_node)
        banner_style = {
            "padding": "6px 10px",
            "borderRadius": "8px",
            "backgroundColor": bg,
            "color": fg,
        }
        return status, options, selected_node, msg, banner_style, power_fig, volt_fig, freq_fig

    # No data yet → placeholders
    placeholder = go.Figure()
    placeholder.update_layout(template="plotly_white", height=300)
    banner_style = {
        "padding": "6px 10px",
        "borderRadius": "8px",
        "backgroundColor": "#fef3c7",
        "color": "#92400e",
    }
    return (
        status,
        options,
        selected_node,
        "Waiting for data… start the replayer to see live updates.",
        banner_style,
        placeholder,
        placeholder,
        placeholder,
    )

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Disable reloader so the MQTT thread runs in the serving process
    app.run(host="127.0.0.1", port=8050, debug=True, use_reloader=False)
