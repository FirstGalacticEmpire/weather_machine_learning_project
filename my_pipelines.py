from sklearn.base import BaseEstimator, TransformerMixin
from re import findall
from geopy.geocoders import Nominatim
import json
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import math

class MaxMinTempDifference(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Warning the result is sometimes negative.
        X["TempDailyDifference"] = X["MaxTemp"] - X["MinTemp"]
        return X


class TempDailyDifference(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Warning the result is sometimes negative.
        X["TempDailyDifference"] = X["Temp3pm"] - X["Temp9am"]
        return X


class WindDailyDifference(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Warning the result is sometimes negative.
        X["WindDailyDifference"] = X["WindSpeed3pm"] - X["WindSpeed9am"]
        return X


class PressureDailyDifference(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Warning the result is sometimes negative.
        X["PressureDailyDifference"] = X["Pressure3pm"] - X["Pressure9am"]
        return X


class HumidityDailyDifference(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Warning the result is sometimes negative.
        X["HumidityDailyDifference"] = X["Humidity3pm"] - X["Humidity9am"]
        return X


class MapLocation(BaseEstimator, TransformerMixin):
    def __init__(self, X, longitude=True, latitude=True, normalize=False) -> None:
        self.longitude = longitude
        self.latitude = latitude
        self.normalize = normalize
        try:
            with open("location_data_for_mapping.json", "r") as file:
                self.dict_of_locations = json.load(file)
                # print(dict_of_locations)
        except FileNotFoundError:
            list_of_locations_from_dataset = list(X["Location"].unique())
            temp = [findall('[A-Z][^A-Z]*', x) for x in list_of_locations_from_dataset]
            search_phrases_for_location = [' '.join(x) + ", Australia" for x in temp]

            geolocator = Nominatim(user_agent="My_own_super_aplication")
            dict_of_locations = {}
            for location, name in zip(search_phrases_for_location, list_of_locations_from_dataset):
                if location == "Pearce R A A F, Australia":
                    location = "Pearce RAAF, Australia"
                loc = geolocator.geocode(location)
                dict_of_locations[name] = {"latitude": loc.latitude, "longitude": loc.longitude}
                print({"latitude": loc.latitude, "longitude": loc.longitude})

            self.dict_of_locations = dict_of_locations
            with open("location_data_for_mapping.json", "w+") as out_file:
                json.dump(dict_of_locations, out_file)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        chosen_dataset = X.copy()

        if self.longitude:
            chosen_dataset.insert(loc=2, column="longitude", value=chosen_dataset["Location"])
            chosen_dataset["longitude"] = chosen_dataset["longitude"].apply(
                lambda x: self.dict_of_locations[x]["longitude"])
            if self.normalize:
                chosen_dataset["longitude"] = chosen_dataset["longitude"].apply(lambda x: x-115.1004768)

        if self.latitude:
            chosen_dataset.insert(loc=2, column="latitude", value=chosen_dataset["Location"])
            chosen_dataset["latitude"] = chosen_dataset["latitude"].apply(
                lambda x: self.dict_of_locations[x]["latitude"])
            if self.normalize:
                chosen_dataset["latitude"] = chosen_dataset["latitude"].apply(lambda x: abs(x+12.46044))

        return chosen_dataset


class NormalizeContinuousFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, columns_to_normalize=None) -> None:
        self.scaler = scaler
        if columns_to_normalize is not None:
            self.columns_to_normalize = columns_to_normalize
        else:
            self.columns_to_normalize = None

    def fit(self, X, y=None):
        if self.columns_to_normalize is not None:
            self.scaler.fit(X[self.columns_to_normalize])
        else:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            self.columns_to_normalize = X.select_dtypes(include=np.number).columns.tolist()
            self.scaler.fit(X[self.columns_to_normalize])
        return self

    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[self.columns_to_normalize] = self.scaler.transform(X_scaled[self.columns_to_normalize])
        return X_scaled


class MeanNANImputer(NormalizeContinuousFeatures):
    def __init__(self, columns_to_normalize=None) -> None:
        super().__init__(SimpleImputer(missing_values=np.nan, strategy='mean'), columns_to_normalize)
        # if columns_to_normalize is not None:
        #     self.columns_to_normalize = columns_to_normalize
        # else:
        #     self.columns_to_normalize = None


class FeaturesFromDate(BaseEstimator, TransformerMixin):
    def __init__(self, drop_date=False) -> None:
        self.drop_date = drop_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Warning the result is sometimes negative.
        copied_dataset = X.copy()
        copied_dataset["Date"] = pd.to_datetime(copied_dataset["Date"])
        copied_dataset['Week_Number'] = pd.to_numeric(copied_dataset['Date'].dt.isocalendar().week, downcast='float')
        copied_dataset['Year'] = pd.to_numeric(copied_dataset['Date'].dt.isocalendar().year, downcast='float')
        if self.drop_date:
            copied_dataset = copied_dataset.drop(columns=["Date"]) #, "Year", "Week_Number"

        return copied_dataset
