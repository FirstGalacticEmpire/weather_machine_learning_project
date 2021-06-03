class RainToNumerical():
    def __init__(self, columns=["RainToday", "RainTomorrow"]):
        self.columns = columns
        self.di = {"No":0, "Yes":1}
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cur = X.copy()
        for col in self.columns:
            cur[col] = cur[col].map(self.di)
        return cur

class WindToDegrees():
    def __init__(self, columns=["WindGustDir", "WindDir9am", "WindDir3pm"]):
        self.columns = columns
        self.di = {}
        directions = ['N','NNE','NE','ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW' ,'NNW']
        deg = 0
        for i in directions:
            self.di[i] = deg
            deg += 22.5
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cur = X.copy()
        for col in self.columns:
            cur[col] = cur[col].map(self.di)
        return cur


