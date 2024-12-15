from abc import ABC, abstractmethod
from typing import List, Any

import pandas as pd


class MissingValuesHandlingStrategy(ABC):
    @abstractmethod
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values for the provided features in the dataframe.
        :param:
            df (pd.DataFrame): The dataframe with missing values.
        :return:
            pd.DataFrame: The dataframe with missing values handled for the provided features.
        """
        pass


class DropMissingValuesStrategy(MissingValuesHandlingStrategy):
    def __init__(self, features: List[str]):
        """
        Initializes a new instance of `DropMissingValuesHandler` with the given axis. If 1 drop columns and if 0 drop rows.
        :param:
            features (List[str]): The features to handle missing values for in the dataframe.
            axis (int): The axis to drop missing values in.
        """
        self._features = features

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops columns or rows with missing values.
        :param:
            df (pd.DataFrame): The dataframe with missing values.
        :return:
            pd.DataFrame: The dataframe with missing values handled for the provided features.
        """
        return df.drop(columns=self._features)


class FillMissingValuesStrategy(MissingValuesHandlingStrategy):
    def __init__(self, features: List[str], method: str = "median", fill_value: Any = None):
        self._features = features
        self._method = method
        self._fill_value = fill_value

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        if self._method == "median":
            df_clean[self._features] = df_clean[self._features].fillna(df_clean[self._features].median())
        elif self._method == "mean":
            df_clean[self._features] = df_clean[self._features].fillna(df_clean[self._features].mean())
        elif self._method == "most_frequent":
            df_clean[self._features] = df_clean[self._features].fillna(df_clean[self._features].mode().iloc[0])
        elif self._method == "constant":
            if self._fill_value is None:
                raise ValueError("The fill_value must be provided when method 'constant' is used.")
            df_clean[self._features] = df_clean[self._features].fillna(self._fill_value)
        else:
            raise ValueError(f"Unsupported method: {self._method}.")

        return df_clean


class MissingValuesHandler:
    def __init__(self, strategy: MissingValuesHandlingStrategy):
        self._strategy = strategy

    def fit(self, X, y=None):
        return self  # No fitting required for missing value handling

    def transform(self, X):
        self._feature_names = X.columns
        return self._strategy.handle_missing_values(X)

    def set_strategy(self, strategy: MissingValuesHandlingStrategy):
        self._strategy = strategy

    def get_params(self, deep=True):
        # Return a dictionary of parameters
        return {"strategy": self._strategy}

    def set_params(self, **params):
        # Set parameters dynamically
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_feature_names_out(self, input_features=None):
        # If input_features are provided, return them unchanged
        return input_features if input_features is not None else self._feature_names
