"""
DER node simulation models.

Purpose
-------
Contains lightweight per-node DER models (DERNode and any specialized helpers)
that produce realistic-ish telemetry samples for different DER types (solar,
wind, battery, EV). Designed for clarity and easy extension, not as detailed
physical models.

How it fits
-----------
This module is the central telemetry generator used by `sim/run_sim.py` and by
the Dash app in `viz/app.py` during local visualization. Later, messages will
be published via MQTT.

Public API
----------
class DERNode:
    - __init__(self, node_id: str, kind: str, base_p_kw: float, seed: Optional[int]=None)
        Create a simulated DER node. `kind` is a string describing the type
        (e.g., 'solar', 'wind', 'battery', 'ev'). `base_p_kw` is nominal capacity.
    - step(self)
        Advance the internal state (time-step) and compute the current telemetry
        sample. Returns a dictionary containing keys such as:
        - timestamp (ISO string)
        - node_id
        - type / kind
        - power_kw (float)
        - voltage (float)
        - frequency_hz (float)
        - possible extras: soc, irradiance, wind_speed, charging
    - get_state(self)
        Returns the current internal state (useful for debugging/visualization).

Notes on Implementation
-----------------------
- The step() method uses simple stochastic processes to generate variation:
  gaussian noise, small drift, diurnal pattern (for solar), gusts (for wind),
  and simple SOC evolution for batteries.
- Designed to be deterministic if a `seed` is provided.

Usage
-----
from sim.nodes import DERNode
n = DERNode("node_solar_1", "solar", base_p_kw=5.0, seed=42)
sample = n.step()

Testing
-------
- Tests should assert that `step()` returns a dictionary with required keys and
  that numeric fields are within expected ranges for a single step.
- For deterministic tests, pass a fixed seed and test for stable outputs.

TODO / Extensions
-----------------
- Add more detailed physics-based models or curve fittings for PV / wind?
- Add event injection hooks (e.g., disconnects, cloud cover shadows).
- Support scenario playback (replay deterministic sequences).
"""

from dataclasses import dataclass
import math, random
from typing import Optional

@dataclass
class DERNode:
    name: str
    kind: str  # "solar" | "battery" | "ev" | "wind"
    base_p_kw: float = 5.0    # nominal active power
    v_pu: float = 1.0         # per-unit voltage
    freq_hz: float = 60.0

    # internal state
    t: int = 0                # seconds since start
    _drift: float = 0.0       # used later for attacks

    def step(self, dt: int = 1, seed: Optional[int] = None):
        """Advance one timestep and update P/V/f with simple, plausible patterns."""
        if seed is not None:
            random.seed(seed + self.t)

        self.t += dt

        # Active power profile (very simple shapes per type)
        if self.kind == "solar":
            # diurnal shape: sine on [0,1], clipped at night; plus noise
            day = (math.sin((self.t % 86400) / 86400 * math.pi) + 0.0)
            p = max(0.0, day) * self.base_p_kw + random.uniform(-0.1, 0.1)
        elif self.kind == "wind":
            # slow varying gusts
            p = self.base_p_kw + 0.5*math.sin(self.t/120) + random.uniform(-0.2, 0.2)
        elif self.kind == "battery":
            # small charge/discharge oscillation
            p = self.base_p_kw + 0.8*math.sin(self.t/300) + random.uniform(-0.1, 0.1)
        elif self.kind == "ev":
            # bursty sessions (on/off)
            on = 1 if (self.t//900) % 2 == 0 else 0
            p = on*self.base_p_kw + random.uniform(-0.1, 0.1)
        else:
            p = self.base_p_kw

        # Voltage & frequency small noise around nominal
        v = 1.0 + random.uniform(-0.01, 0.01)
        f = 60.0 + random.uniform(-0.02, 0.02)

        # assign
        self.v_pu, self.freq_hz, self.base_p_kw = round(v,4), round(f,4), round(p,4)

        return {
            "id": self.name,
            "P": self.base_p_kw,
            "V": self.v_pu,
            "F": self.freq_hz,
            "t": self.t
        }
