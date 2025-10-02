"""
Dash-based visualization / local simulator.

Purpose
-------
A minimal Dash app that runs a local instance of the DER simulation and
renders live plots and basic status indicators. In the initial prototype the
visualization drives the simulation locally; in the full pipeline the viz will
subscribe to MQTT topics instead and become a consumer-only visualization layer.

Key behaviours
-------------
- Creates the same types of DERNode objects used by sim/run_sim.py.
- Advances the node states on a Dash interval (default 1s) and updates plots.
- Shows recent history for a selected node and a simple textual alert when
  certain thresholds are exceeded.

How to run
----------
# From project root:
python -m viz.app

Then open browser at http://127.0.0.1:8050

Public functions / objects
--------------------------
- create_app() -> dash.Dash (optional)
    If present, helps unit-testing the layout and callbacks without running the
    server. If your code already instantiates app at module-level, consider
    exporting `app` and `server` for deployability.

Common changes for demo
-----------------------
- Add a command-line flag `--mqtt` to switch between local simulation and MQTT
  subscription mode.
- Add a configuration option to select nodes to display and history length.

Testing
-------
- Dash UI tests are typically end-to-end (Selenium or Playwright). For unit
  tests, assert that callbacks produce correct transformed data given
  deterministic node step outputs (use fixed seeds).
"""

import dash
from dash import dcc, html, Output, Input, State
import plotly.graph_objs as go
import pandas as pd
import time
from collections import deque

from sim.nodes import DERNode
from sim.log import Log

# Config
NODES = [
    DERNode("solar1", "solar", base_p_kw=5.0),
    DERNode("wind1",  "wind",  base_p_kw=4.0),
    DERNode("bat1",   "battery", base_p_kw=2.5),
    DERNode("ev1",    "ev",    base_p_kw=7.0),
]
SAMPLE_INTERVAL_MS = 1000  # dashboard update every 1s
HISTORY_POINTS = 120  # seconds worth of history on the chart

# runtime state
log = Log(NODES, scenario="live")
# keep recent history per node for charting
history = {n.name: deque(maxlen=HISTORY_POINTS) for n in NODES}
time_history = deque(maxlen=HISTORY_POINTS)

# Dash app
app = dash.Dash(__name__)
app.title = "DERMS Live Visualizer"

# build simple node cards (icon + text)
def node_card(node):
    return html.Div(
        id=f"card-{node.name}",
        children=[
            html.Div(node.name, style={"fontWeight":"bold"}),
            html.Div(id=f"val-{node.name}", children="P: -- kW | V: -- pu", style={"fontSize":"14px"})
        ],
        style={
            "border":"1px solid #ddd", "borderRadius":"6px", "padding":"8px",
            "width":"220px", "textAlign":"center", "background":"#fff"
        }
    )

app.layout = html.Div([
    html.H2("DERMS Live Visualizer", style={"color":"#1B3A5F"}),
    html.Div([
        # left: node cards
        html.Div([node_card(n) for n in NODES], style={"display":"flex","gap":"12px","flexWrap":"wrap"}),
        # right: controls + chart
        html.Div([
            html.Label("Select node:"),
            dcc.Dropdown(id="node-select", options=[{"label":n.name,"value":n.name} for n in NODES], value=NODES[0].name),
            dcc.Graph(id="live-chart", style={"height":"360px"}),
            html.Div(id="alert-box", style={"marginTop":"8px","fontWeight":"bold"})
        ], style={"width":"100%","marginTop":"12px"})
    ], style={"display":"grid","gridTemplateColumns":"1fr","gap":"18px"}),
    # interval updater
    dcc.Interval(id="interval", interval=SAMPLE_INTERVAL_MS, n_intervals=0),
    # hidden store for history if needed
    dcc.Store(id="store-history")
], style={"fontFamily":"Arial, Helvetica, sans-serif","padding":"18px", "background":"#F8F9FA"})

# Callback: update sim and UI each tick
@app.callback(
    Output("live-chart","figure"),
    Output("alert-box","children"),
    [Input("interval","n_intervals"), Input("node-select","value")]
)
def update_live(n_intervals, selected_node):
    # advance simulation one step for all nodes
    ts = pd.Timestamp.utcnow().isoformat()
    for n in NODES:
        n.step(dt=1)
    log.add_row(ts)

    # record history
    time_history.append(ts)
    for n in NODES:
        history[n.name].append(n.base_p_kw)

    # build line chart for selected node
    y = list(history[selected_node])
    x = list(time_history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=f"{selected_node} P (kW)",
                             line=dict(color="#4A90E2")))
    fig.update_layout(margin=dict(l=40,r=20,t=30,b=40), xaxis_title="Time", yaxis_title="Active Power (kW)",
                      template="simple_white")

    # build alert summary (simple rule: power below zero or big jump)
    alerts = []
    for n in NODES:
        val = n.base_p_kw
        # example alert rule: abnormal negative power or > 2x base
        if val < -1.0 or val > (n.base_p_kw*3 + 2):  # this is illustrative
            alerts.append(f"ALERT {n.name}: unusual power {val:.2f} kW")
    alert_text = "; ".join(alerts) if alerts else "All nodes nominal."

    # update node cards via client-side? Simple approach: update via dcc.Graph hover (we will just return chart + alert)
    return fig, alert_text

if __name__ == "__main__":
    app.run(debug=True, port=8050)  # Dash 2.16+ API
