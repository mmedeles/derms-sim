class RF_Classifier:
    def __init__(self):
        #load model from file
        pass
    def classify(self,x) -> bool:
        #y=self.model predict(x)
        if x[0]>60.00:
            return True
        else:
            return False

#class XGB_Classifer:
#class LSTM_Classifier: