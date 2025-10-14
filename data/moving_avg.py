import math
from typing import List, Optional

class Node_MA:
    """
    Rolling window moving-average & moving-std tracker.
    """
    def __init__(self, n: int = 60, ddof: int = 1):
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

    def average(self) -> float:
        return sum(self.window) / len(self.window) if self.window else 0.0

    def std(self) -> float:
        n = len(self.window)
        if n <= self.ddof or n == 0:
            return 0.0
        m = self.average()
        var = sum((x - m) ** 2 for x in self.window) / (n - self.ddof)
        return math.sqrt(var)
