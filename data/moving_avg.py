

class Node_MA:
    def __init__(self,n):
        self.n=n #length of tracking
        self.last_n = []
    def average(self):
        return sum(self.last_n) / len(self.last_n)
    def update(self,new_value):
        self.last_n.append(new_value)
        if len(self.last_n) > self.n:
            self.last_n.pop(0)
        return self.average()