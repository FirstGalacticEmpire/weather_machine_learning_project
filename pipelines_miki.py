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

class RemoveOutliers():
    def __init__(self, scope=3.5): #scope is a std multiplier - the smaller it is the more values will be treated as outliers
        self.scope=scope
        self.columns = []
        self.MeanAndStd = {}
            
    def fit(self, X, y=None):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.columns = X.select_dtypes(include=np.number).columns.tolist()
        for col in self.columns:
            self.MeanAndStd[col] = [X[col].mean(), X[col].std()]
        return self
    
    def transform(self, X):
        #print(len(X.Location.unique()))
        cur = X.copy()
        for col in self.columns:
            mean, std = self.MeanAndStd[col]
            cutOff = std * self.scope
            cur = cur[cur[col] >= mean - cutOff & cur[col] <= mean + cutOff & ~cur[col].isnull()]
        #print(len(cur.Location.unique()))
        print("Removed: {}".format(len(X[col]) - len(cur[col])))
        #print(len(cur[cur["RainTomorrow"] == "No"]), len(cur[cur["RainTomorrow"] == "Yes"]))
        #print("NAs:", cur.isna().sum())
        return cur
