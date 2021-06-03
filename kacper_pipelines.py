import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import PolynomialFeatures

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        #print(X.columns)
        #print(X.head())
        #print(X.isna().sum()/X.shape[0] * 100)
        return X.drop(columns=self.columns)

class DropRowsWithNAInColums(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.columns is None:
            return X.dropna(subset=X.columns)
        return X.dropna(subset=self.columns) 


class DropRowsWithMoreThanXNA(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.dropna(thresh=len(X.columns) - self.threshold)


class DropColumnsAbovePercentNA(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
    def fit(self, X, y=None):
        self.columns = X.columns[X.isna().sum()/X.shape[0] * 100 <= self.threshold]     
        return self
    def transform(self, X):
        return X[self.columns]


class PolynomialSubset(BaseEstimator, TransformerMixin):
    def __init__(self, columns, degree = 2):
        self.columns = columns
        self.degree = degree
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        cur = X.copy()
        subset = cur[self.columns]
        cur.drop(columns = self.columns, inplace=True)
        poly = PolynomialFeatures(degree = self.degree)
        poly_subset = poly.fit_transform(subset)
        poly_subset = pd.DataFrame(poly_subset, columns = poly.get_feature_names(subset.columns), index = cur.index)
        poly_subset.drop(columns=['1'], inplace = True)
        return pd.concat([cur,poly_subset], axis = 1)

