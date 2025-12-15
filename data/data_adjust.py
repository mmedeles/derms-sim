import numpy as np
from random import randint,seed
class data_adjust:
    def __init__(self, magnitude, method, period=360,sigma=None):
        self.M=magnitude
        self.t=-1
        self.method = {"gaussian":self.gaussian_adjust,"linear":self.linear_adjust}[method]
        self.period=period
        seed(0)
        if sigma:
            self.sigma=sigma
        else:
            self.sigma=(self.period / 2) / np.sqrt(2 * np.log(1 / 0.01))
    def adjust(self):
        self.t+=1
        if self.t>=self.period:
            self.t=randint(-1000,-10)
        if self.t<0:
            return 0
        return self.method()
    
    def gaussian_adjust(self):
        return self.M * np.exp(-((self.t-(self.period/2))**2) / (2*self.sigma**2))
    def linear_adjust(self):
        return self.t*self.M
    
#g = data_adjust(1,"gaussian")    
#for i in range(150):
    #print(g.adjust())