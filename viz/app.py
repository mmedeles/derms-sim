"""
Dash visualization that CONSUMES live telemetry over MQTT.

Supports BOTH topic shapes (backward compatible):

1) Multi-stream (Spring 2026 Phase 1):
   - Telemetry: derms/<stream_id>/<node_id>/telemetry
   - Anomaly:   derms/<stream_id>/<node_id>/anomaly  (optional)

2) Legacy (single-stream):
   - Telemetry: derms/<node_id>/telemetry
   - Anomaly:   derms/<node_id>/anomaly

What it shows:
  * Facility Overview (all active streams/nodes) with last-seen telemetry + vote summary
  * Detailed view for a selected stream/node:
      - Power (kW), Voltage (V), Frequency (Hz)
      - Frequency overlays: Moving Avg (dotted) and ±k·STD shaded band (if present)

Key usability fix:
  - Default x-axis uses *ingest time* so the plots behave like a live strip-chart
    even if the replayed CSV timestamps are old or looping.

Env:
  MQTT_HOST             (default: 127.0.0.1)
  MQTT_PORT             (default: 1883)
  HISTORY_POINTS        (default: 2400)      # buffer length per node
  PLOT_POINTS           (default: 600)       # max points plotted
  MA_FIELD              (default: moving_avg_frequency_hz)
  STD_FIELD             (default: moving_std_frequency_hz)
  STD_BAND_FACTOR       (default: 1.0)
  TIME_MODE             (default: ingest)    # ingest | payload (UI can override)
  STRIP_WINDOW_SEC      (default: 120)       # x-axis window for strip chart
  STALE_SEC             (default: 5)         # "stale" if last ingest older than this
"""

from __future__ import annotations

import os
import json
import threading
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, List

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

HISTORY_POINTS = int(os.getenv("HISTORY_POINTS", "2400"))
PLOT_POINTS = int(os.getenv("PLOT_POINTS", "600"))

MA_FIELD = os.getenv("MA_FIELD", "moving_avg_frequency_hz")
STD_FIELD = os.getenv("STD_FIELD", "moving_std_frequency_hz")
STD_BAND_FACTOR = float(os.getenv("STD_BAND_FACTOR", "1.0"))

DEFAULT_TIME_MODE = os.getenv("TIME_MODE", "ingest").strip().lower()  # ingest | payload
DEFAULT_STRIP_WINDOW_SEC = int(os.getenv("STRIP_WINDOW_SEC", "120"))
STALE_SEC = int(os.getenv("STALE_SEC", "5"))

# stream/node composite key -> deque of telemetry dicts
BUFFER: Dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_POINTS))
ANOMALIES: Dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_POINTS))
KNOWN_NODES = set()

MQTT_CONNECTED = False


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _topic_parts(topic: str) -> List[str]:
    return [p for p in (topic or "").split("/") if p]


def _extract_stream_node(topic: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (stream_id, node_id) with backward compatibility.

    Multi-stream topic:
      derms/<stream_id>/<node_id>/telemetry
    Legacy topic:
      derms/<node_id>/telemetry
    """
    parts = _topic_parts(topic)

    # payload values first if present
    node_id = str(payload.get("node_id") or "unknown")
    stream_id = str(payload.get("stream_id") or "")

    # topic parsing fallback
    # Expected: ["derms", "<stream_id>", "<node_id>", "telemetry"]
    if len(parts) >= 4 and parts[0] == "derms" and parts[-1] in ("telemetry", "anomaly"):
        if not stream_id:
            stream_id = parts[1]
        # prefer topic's node_id if payload missing/unknown
        if (node_id == "unknown") and parts[2]:
            node_id = parts[2]

    # Legacy: ["derms", "<node_id>", "telemetry"]
    if len(parts) == 3 and parts[0] == "derms" and parts[-1] in ("telemetry", "anomaly"):
        if not stream_id:
            stream_id = "stream_1"
        if node_id == "unknown":
            node_id = parts[1]

    if not stream_id:
        stream_id = "stream_1"

    return stream_id, node_id


def _node_key(stream_id: str, node_id: str) -> str:
    return f"{stream_id}/{node_id}"


def _format_vote_summary(d: Dict[str, Any]) -> str:
    flags = []
    for k in ("is_anom_rf", "is_anom_lstm", "is_anom_svm", "is_anom_xgb"):
        v = d.get(k)
        try:
            v = int(v)
        except Exception:
            v = 0
        flags.append(v)
    votes = sum(flags)
    return f"{votes}/4 votes"


def _status_color(votes: int) -> Tuple[str, str]:
    # background, foreground
    if votes >= 3:
        return "#fee2e2", "#991b1b"   # red-ish
    if votes == 2:
        return "#ffedd5", "#9a3412"   # orange-ish
    return "#ecfdf5", "#065f46"       # green-ish


def _line_figure(title: str, x, y, y_title: str, x_range: Optional[Tuple[datetime, datetime]] = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=title))
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=10, t=70, b=40),
        height=280,
        xaxis=dict(
            tickformat="%H:%M:%S",
            tickmode="auto",
            nticks=8,
            showgrid=True,
            range=x_range if x_range else None,
        ),
        yaxis_title=y_title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        uirevision="keep",
    )
    return fig


# ------------------------------------------------------------------------------
# MQTT subscriber
# ------------------------------------------------------------------------------
def _start_mqtt():
    global MQTT_CONNECTED

    if mqtt is None:
        print("[viz] paho-mqtt not available. Install: pip install paho-mqtt")
        return

    client = mqtt.Client(client_id="viz_ui", protocol=mqtt.MQTTv311)

    def on_connect(_c, _u, _flags, rc):
        nonlocal client
        MQTT_CONNECTED = (rc == 0)
        if rc != 0:
            print(f"[viz] MQTT connect failed rc={rc} host={MQTT_HOST}:{MQTT_PORT}")
            return

        # Subscribe to BOTH shapes (multi-stream + legacy)
        client.subscribe("derms/+/+/telemetry")
        client.subscribe("derms/+/+/anomaly")
        client.subscribe("derms/+/telemetry")
        client.subscribe("derms/+/anomaly")
        print(
            f"[viz] Subscribed to MQTT at {MQTT_HOST}:{MQTT_PORT} "
            f"(derms/+/+/telemetry, derms/+/+/anomaly, legacy derms/+/telemetry)"
        )

    def on_disconnect(_c, _u, rc):
        MQTT_CONNECTED = False
        if rc != 0:
            print(f"[viz] MQTT disconnected rc={rc}")

    def on_message(_c, _u, msg):
        topic = msg.topic or ""
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            if not isinstance(data, dict):
                return
        except Exception:
            return

        stream_id, node_id = _extract_stream_node(topic, data)
        key = _node_key(stream_id, node_id)

        # Always add ingest timestamp (this is what makes strip charts behave)
        data["_ingest_ts"] = _utc_now().isoformat().replace("+00:00", "Z")
        data.setdefault("stream_id", stream_id)
        data.setdefault("node_id", node_id)

        if topic.endswith("/telemetry"):
            # allow telemetry even if payload has no timestamp; dashboard can use ingest time
            if not data.get("timestamp"):
                data["timestamp"] = data["_ingest_ts"]
            BUFFER[key].append(data)
            KNOWN_NODES.add(key)
        elif topic.endswith("/anomaly"):
            ANOMALIES[key].append(data)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    try:
        client.connect(MQTT_HOST, MQTT_PORT, 60)
    except Exception as e:
        print(f"[viz] MQTT connect error: {e}")
        return

    client.loop_forever()


threading.Thread(target=_start_mqtt, daemon=True).start()


# ------------------------------------------------------------------------------
# Dash app
# ------------------------------------------------------------------------------
app = dash.Dash(__name__)
server = app.server
app.title = "DERMS Live Dashboard"


def _pill(text: str, bg: str, fg: str):
    return html.Span(
        text,
        style={
            "display": "inline-block",
            "padding": "3px 8px",
            "borderRadius": "999px",
            "backgroundColor": bg,
            "color": fg,
            "fontSize": "12px",
            "fontWeight": 600,
            "border": "1px solid rgba(0,0,0,0.08)",
        },
    )


app.layout = html.Div(
    style={
        "fontFamily": "Inter, system-ui, sans-serif",
        "padding": "16px",
        "maxWidth": "1400px",
        "margin": "0 auto",
    },
    children=[
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "baseline", "gap": "12px"},
            children=[
                html.Div(
                    children=[
                        html.H2("DERMS Live Dashboard", style={"margin": 0}),
                        html.Div(
                            "Multi-stream proof-of-concept: 3–4 independent streams publishing to MQTT.",
                            style={"color": "#6b7280", "marginTop": "4px"},
                        ),
                    ]
                ),
                html.Div(id="clock", style={"color": "#6b7280"}),
            ],
        ),

        html.Div(style={"height": "10px"}),

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

        # Controls
        html.Div(
            style={
                "display": "flex",
                "gap": "12px",
                "alignItems": "center",
                "flexWrap": "wrap",
                "marginBottom": "10px",
            },
            children=[
                html.Div(
                    style={"minWidth": "240px"},
                    children=[
                        html.Label("Selected Stream/Node", style={"fontWeight": 600}),
                        dcc.Dropdown(
                            id="node-select",
                            options=[],
                            value=None,
                            placeholder="Waiting for data…",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "180px"},
                    children=[
                        html.Label("Time Base", style={"fontWeight": 600}),
                        dcc.Dropdown(
                            id="time-mode",
                            options=[
                                {"label": "Ingest time (recommended)", "value": "ingest"},
                                {"label": "Payload timestamp", "value": "payload"},
                            ],
                            value=DEFAULT_TIME_MODE if DEFAULT_TIME_MODE in ("ingest", "payload") else "ingest",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "320px", "flex": "1 1 320px"},
                    children=[
                        html.Label("Strip Window (seconds)", style={"fontWeight": 600}),
                        dcc.Slider(
                            id="strip-window",
                            min=30,
                            max=600,
                            step=10,
                            value=DEFAULT_STRIP_WINDOW_SEC,
                            marks={30: "30", 120: "120", 300: "300", 600: "600"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Div(id="anomaly-banner", style={"fontWeight": 600}),
            ],
        ),

        # Facility overview (all nodes)
        html.Div(
            children=[
                html.H4("Facility Overview", style={"margin": "8px 0 6px 0"}),
                html.Div(
                    id="overview",
                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                ),
            ]
        ),

        html.Div(style={"height": "12px"}),

        # Detailed graphs (selected node)
        dcc.Graph(id="power-graph", config={"displayModeBar": False}),
        dcc.Graph(id="voltage-graph", config={"displayModeBar": False}),
        dcc.Graph(id="frequency-graph", config={"displayModeBar": False}),

        dcc.Interval(id="tick", interval=1000, n_intervals=0),
    ],
)


# ------------------------------------------------------------------------------
# Data extraction
# ------------------------------------------------------------------------------
def _get_series_for_node(node_key: str, time_mode: str, max_points: int = PLOT_POINTS):
    if not node_key or node_key not in BUFFER or len(BUFFER[node_key]) == 0:
        return [], [], [], [], [], [], [], [], [], [], []

    items = list(BUFFER[node_key])[-max_points:]
    parsed: List[Tuple[datetime, Dict[str, Any]]] = []

    for d in items:
        if time_mode == "ingest":
            t = _parse_ts(d.get("_ingest_ts", ""))
        else:
            t = _parse_ts(d.get("timestamp", ""))
        if t is not None:
            parsed.append((t, d))

    if not parsed:
        return [], [], [], [], [], [], [], [], [], [], []

    times = [t for (t, _) in parsed]
    p_kw = [_safe_float(x.get("power_kw"), 0.0) for _, x in parsed]
    v = [_safe_float(x.get("voltage"), 0.0) for _, x in parsed]
    f_hz = [_safe_float(x.get("frequency_hz"), 0.0) for _, x in parsed]

    ma_vals, std_vals, adjusted_vals = [], [], []
    is_anom_rf, is_anom_lstm, is_anom_svm, is_anom_xgb = [], [], [], []
    for _, x in parsed:
        mv = _safe_float(x.get(MA_FIELD))
        ma_vals.append(mv)

        sv = _safe_float(x.get(STD_FIELD))
        std_vals.append(sv)

        adjusted_vals.append(_safe_float(x.get("Hz_adjusted")))

        def _as_int(val):
            try:
                return int(val)
            except Exception:
                return 0

        is_anom_rf.append(_as_int(x.get("is_anom_rf")))
        is_anom_lstm.append(_as_int(x.get("is_anom_lstm")))
        is_anom_svm.append(_as_int(x.get("is_anom_svm")))
        is_anom_xgb.append(_as_int(x.get("is_anom_xgb")))

    return times, p_kw, v, f_hz, ma_vals, std_vals, adjusted_vals, is_anom_rf, is_anom_lstm, is_anom_svm, is_anom_xgb


def _recent_anomaly_label(node_key: str):
    if node_key in ANOMALIES and len(ANOMALIES[node_key]) > 0:
        recent = list(ANOMALIES[node_key])[-50:]
        if any(bool(a.get("anomaly")) for a in recent):
            return ("Anomaly detected in recent samples", "#fee2e2", "#991b1b")
    return ("No recent anomalies", "#ecfdf5", "#065f46")


def _overview_cards(now: datetime) -> List[html.Div]:
    cards: List[html.Div] = []

    for key in sorted(KNOWN_NODES):
        if key not in BUFFER or len(BUFFER[key]) == 0:
            continue

        d = BUFFER[key][-1]
        ingest = _parse_ts(d.get("_ingest_ts", "")) or now
        age_s = (now - ingest).total_seconds()

        stream_id = str(d.get("stream_id", "stream_1"))
        node_id = str(d.get("node_id", "unknown"))

        power_kw = _safe_float(d.get("power_kw"))
        voltage = _safe_float(d.get("voltage"))
        freq = _safe_float(d.get("frequency_hz"))
        votes_txt = _format_vote_summary(d)

        flags = []
        for k in ("is_anom_rf", "is_anom_lstm", "is_anom_svm", "is_anom_xgb"):
            try:
                flags.append(int(d.get(k) or 0))
            except Exception:
                flags.append(0)
        votes = sum(flags)
        bg, fg = _status_color(votes)

        stale = age_s > STALE_SEC
        stale_pill = _pill("STALE" if stale else "LIVE", "#fef3c7" if stale else "#e0f2fe", "#92400e" if stale else "#075985")

        cards.append(
            html.Div(
                style={
                    "width": "260px",
                    "border": "1px solid #e5e7eb",
                    "borderRadius": "10px",
                    "padding": "10px",
                    "backgroundColor": "white",
                    "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                        children=[
                            html.Div(
                                children=[
                                    html.Div(f"{stream_id}", style={"fontSize": "12px", "color": "#6b7280"}),
                                    html.Div(f"{node_id}", style={"fontWeight": 700, "fontSize": "16px"}),
                                ]
                            ),
                            stale_pill,
                        ],
                    ),
                    html.Div(style={"height": "8px"}),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "6px"},
                        children=[
                            html.Div(children=[html.Div("Power", style={"color": "#6b7280", "fontSize": "12px"}),
                                               html.Div(f"{power_kw:.3f} kW" if power_kw is not None else "—",
                                                        style={"fontWeight": 600})]),
                            html.Div(children=[html.Div("Voltage", style={"color": "#6b7280", "fontSize": "12px"}),
                                               html.Div(f"{voltage:.2f} V" if voltage is not None else "—",
                                                        style={"fontWeight": 600})]),
                            html.Div(children=[html.Div("Freq", style={"color": "#6b7280", "fontSize": "12px"}),
                                               html.Div(f"{freq:.4f} Hz" if freq is not None else "—",
                                                        style={"fontWeight": 600})]),
                            html.Div(children=[html.Div("Votes", style={"color": "#6b7280", "fontSize": "12px"}),
                                               _pill(votes_txt, bg, fg)]),
                        ],
                    ),
                    html.Div(style={"height": "8px"}),
                    html.Div(
                        f"Last ingest: {ingest.strftime('%H:%M:%S')}Z  •  Age: {age_s:.1f}s",
                        style={"fontSize": "12px", "color": "#6b7280"},
                    ),
                ],
            )
        )

    if not cards:
        cards.append(
            html.Div(
                "Waiting for telemetry… start the replayer and confirm MQTT topics are publishing.",
                style={
                    "padding": "12px",
                    "borderRadius": "10px",
                    "border": "1px dashed #d1d5db",
                    "color": "#6b7280",
                },
            )
        )
    return cards


# ------------------------------------------------------------------------------
# Callback
# ------------------------------------------------------------------------------
@app.callback(
    Output("clock", "children"),
    Output("mqtt-status", "children"),
    Output("node-select", "options"),
    Output("node-select", "value"),
    Output("anomaly-banner", "children"),
    Output("anomaly-banner", "style"),
    Output("overview", "children"),
    Output("power-graph", "figure"),
    Output("voltage-graph", "figure"),
    Output("frequency-graph", "figure"),
    Input("tick", "n_intervals"),
    State("node-select", "value"),
    State("time-mode", "value"),
    State("strip-window", "value"),
)
def update_ui(_n, selected_node, time_mode, strip_window_sec):
    now = _utc_now()
    clock = f"UTC now: {now.strftime('%Y-%m-%d %H:%M:%S')}Z"

    mqtt_ok = mqtt is not None
    have_data = len(KNOWN_NODES) > 0
    conn = "OK" if MQTT_CONNECTED else "Disconnected"
    status = (
        f"MQTT host: {MQTT_HOST}:{MQTT_PORT}  •  Client: {'OK' if mqtt_ok else 'Unavailable'}  •  Broker: {conn}  •  "
        f"Nodes observed: {', '.join(sorted(KNOWN_NODES)) if have_data else '— waiting —'}"
    )

    options = [{"label": k, "value": k} for k in sorted(KNOWN_NODES)]
    if selected_node is None and options:
        selected_node = options[0]["value"]

    overview_children = _overview_cards(now)

    placeholder = _line_figure("", [], [], "", x_range=None)
    placeholder.update_layout(showlegend=False)

    if not selected_node or selected_node not in BUFFER or len(BUFFER[selected_node]) == 0:
        banner_style = {"padding": "6px 10px", "borderRadius": "8px", "backgroundColor": "#fef3c7", "color": "#92400e"}
        return (
            clock,
            status,
            options,
            selected_node,
            "Waiting for data… start the replayer to see live updates.",
            banner_style,
            overview_children,
            placeholder,
            placeholder,
            placeholder,
        )

    time_mode = (time_mode or DEFAULT_TIME_MODE).strip().lower()
    strip_window_sec = int(strip_window_sec or DEFAULT_STRIP_WINDOW_SEC)

    (
        times,
        p_kw,
        v,
        f_hz,
        ma_vals,
        std_vals,
        adjusted_vals,
        is_anom_rf,
        is_anom_lstm,
        is_anom_svm,
        is_anom_xgb,
    ) = _get_series_for_node(selected_node, time_mode=time_mode, max_points=PLOT_POINTS)

    if not times:
        banner_style = {"padding": "6px 10px", "borderRadius": "8px", "backgroundColor": "#fef3c7", "color": "#92400e"}
        return (
            clock,
            status,
            options,
            selected_node,
            "Waiting for parsable timestamps…",
            banner_style,
            overview_children,
            placeholder,
            placeholder,
            placeholder,
        )

    t_end = times[-1]
    t_start = t_end - timedelta(seconds=strip_window_sec)
    x_range = (t_start, t_end)

    power_fig = _line_figure("Power (kW)", times, p_kw, "kW", x_range=x_range)
    volt_fig = _line_figure("Voltage (V)", times, v, "V", x_range=x_range)
    freq_fig = _line_figure("Frequency (Hz)", times, f_hz, "Hz", x_range=x_range)

    if any(val is not None for val in ma_vals):
        freq_fig.add_trace(go.Scatter(x=times, y=ma_vals, mode="lines", name="Moving Avg", line=dict(dash="dot")))

    if any(val is not None for val in adjusted_vals):
        symbol_map = {
            0: ("circle", 6, "blue"),
            1: ("star", 10, "blue"),
            2: ("hexagon", 13, "red"),
            3: ("diamond", 16, "red"),
            4: ("x", 20, "red"),
        }

        pred_sums = [int(a) + int(b) + int(c) + int(d) for a, b, c, d in zip(is_anom_rf, is_anom_lstm, is_anom_svm, is_anom_xgb)]
        symbols, sizes, colors = zip(*(symbol_map.get(s, ("circle", 6, "blue")) for s in pred_sums))

        freq_fig.add_trace(
            go.Scatter(
                x=times,
                y=adjusted_vals,
                mode="markers+lines",
                name="Adjusted Values",
                marker=dict(symbol=symbols, size=sizes, color=colors, line=dict(width=1)),
            )
        )

    if STD_BAND_FACTOR > 0 and any(s is not None for s in std_vals) and any(m is not None for m in ma_vals):
        upper = []
        lower = []
        for m, s in zip(ma_vals, std_vals):
            if m is None or s is None:
                upper.append(None)
                lower.append(None)
            else:
                upper.append(m + STD_BAND_FACTOR * s)
                lower.append(m - STD_BAND_FACTOR * s)

        freq_fig.add_trace(go.Scatter(x=times, y=upper, mode="lines", name=f"+{STD_BAND_FACTOR}·STD", line=dict(width=0), showlegend=False))
        freq_fig.add_trace(
            go.Scatter(
                x=times,
                y=lower,
                mode="lines",
                name=f"-{STD_BAND_FACTOR}·STD",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(59,130,246,0.15)",
                showlegend=False,
            )
        )

    msg, bg, fg = _recent_anomaly_label(selected_node)
    banner_style = {"padding": "6px 10px", "borderRadius": "8px", "backgroundColor": bg, "color": fg}

    return clock, status, options, selected_node, msg, banner_style, overview_children, power_fig, volt_fig, freq_fig


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True, use_reloader=False)
