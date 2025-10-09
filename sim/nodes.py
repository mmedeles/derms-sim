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
import numpy as np
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
class Cloud:
    def __init__(self):
        rng=np.random.default_rng(0)
        mu=5
        sig=4
        self.duration = int(rng.lognormal(mu,sig))
        self.fractional = max(min(np.random.normal(.5,.25),0.99),0.05)
    def step(self):
        self.duration-=1
        if self.duration==0:
            return self.fractional,True
        else:
            return self.fractional,False
class SolarNode(DERNode):
    def __init__(self,name):
        super().__init__(name,"solar")
        self.sunrise=23750 #approx
        self.sunset=62700
        self.cloud=None
        self.day_cloudiness=200 #100 very cloud, 1000 not too cloudy 10000+ very clear
        
    def W_from_sec(self,sec): #this function predicts W based on time of day based on solar data from prakash #seconds must start at 0 at a midnight
        sec=sec%86400
        if sec<self.sunrise or sec>self.sunset:
            return 0
        
        peak=9333 #max of average day line
        return max(peak*(math.sin(math.pi*(sec-self.sunrise)/(self.sunset-self.sunrise))**1.7)  + np.random.normal(0,8), 0)
    
    def make_cloud(self):
        if random.randint(1,self.day_cloudiness) ==1:
            self.cloud=Cloud()
    def change_cloudiness(self):
        roll = random.randint(0,9)
        if roll<5:
            self.day_cloudiness=20000
        elif roll<6:
            self.day_cloudiness=10000
        elif roll<7:
            self.day_cloudiness=5000
        elif roll<8:
            self.day_cloudiness=1000
        elif roll<9:
            self.day_cloudiness=200    
        elif roll<10:
            self.day_cloudiness=100
    def step(self,dt: int =1):
        """Advance one timestep and update P/V/f with simple, plausible patterns."""
        self.t += dt
        
        if random.randint(0,45000) == 0: #~ twice a day change to a different level of cloudiness
            self.change_cloudiness()
        
        
        p = self.W_from_sec(self.t)
        v = 1.0 + random.uniform(-0.01, 0.01)
        #PhVphA is mean 239.6 std 2.38, somewhat gaussian
        
        if self.t%86400>self.sunrise and self.t%86400<self.sunset: #if its daytime
            f = 59.996 + np.random.normal(0,0.017)
        else:
            f=0
            
        if self.cloud==None:
            self.make_cloud()
        else:
            frac,expire = self.cloud.step()
            if expire:
                self.cloud=None
            p=p*frac
            
        # assign
        self.v_pu, self.freq_hz, self.base_p_kw = round(v,4), round(f,4), round(p,4)
    
        return {
                "id": self.name,
                "P": self.base_p_kw,
                "V": self.v_pu,
                "F": self.freq_hz,
                "t": self.t
            }          
