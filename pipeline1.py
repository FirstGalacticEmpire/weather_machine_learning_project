import pandas as pd
from my_pipelines import *
from sklearn import preprocessing
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

FILE_PATH = './weatherAUS.csv'
rain_data = pd.read_csv(FILE_PATH)

if __name__ == "__main__":
    print(rain_data.head(3))
    #print(rain_data.dtypes)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_columns = rain_data.select_dtypes(include=np.number).columns.tolist()

    print(numeric_columns)

    # transformer = NormalizeContinuousFeatures(preprocessing.StandardScaler(), None)
    # rain_data = transformer.fit_transform(rain_data)
    # print(rain_data.head(3))


    # columns_to_normalize = ["MinTemp", "MaxTemp"]
    #
    # sub_set = rain_data[columns_to_normalize]
    # print(sub_set.head(1))
    # scaler = preprocessing.StandardScaler().fit(sub_set)
    # print(scaler.mean_)
    #
    # print(rain_data.head(5))
