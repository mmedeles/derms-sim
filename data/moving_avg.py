import math
from typing import List, Optional

class Node_MA:
    """
    Rolling window moving-average & moving-std tracker.
    """
    def __init__(self, n: int = 240, ddof: int = 1):
        self.n = max(1, int(n))
        self.ddof = 0 if ddof is None else int(ddof)
        self.window: List[float] = []

    def update(self, new_value: Optional[float]) -> float:
        try:
            if new_value is None:
                return self.average()
            v = float(new_value)
        except Exception:
            return self.average()
        self.window.append(v)
        if len(self.window) > self.n:
            self.window.pop(0)
        return self.average()

    def average(self,n=60) -> float:
        return sum(self.window[-n:]) / len(self.window[-n:]) if self.window else 0.0

    def std(self,n_=60) -> float:
        window_slice=self.window[-n_:]
        n = len(window_slice)
        if n <= self.ddof or n == 0:
            return 0.0
        m = sum(window_slice) / n
        var = sum((x - m) ** 2 for x in window_slice) / (n - self.ddof)
        return math.sqrt(var)
