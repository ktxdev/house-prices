from abc import ABC, abstractmethod
from typing import List, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
    def __init__(self, features: List[str], method: str = "median", grouping_feature: str = None,
                 fill_value: Any = None):
        self._features = features
        self._method = method
        self._grouping_feature = grouping_feature
        self._fill_value = fill_value

    def _get_fill_function(self):
        if self._method == "median":
            return lambda x: x.median()
        elif self._method == "mean":
            return lambda x: x.mean()
        elif self._method == "most_frequent":
            return lambda x: x.mode().iloc[0] if not x.mode().empty else None
        elif self._method == "constant":
            if self._fill_value is None:
                raise ValueError("The fill_value must be provided when method 'constant' is used.")
            return lambda x: x.fillna(self._fill_value)
        else:
            raise ValueError(f"Unsupported method: {self._method}.")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        # Get the fill action
        fill_function = self._get_fill_function()
        for feature in self._features:
            df_clean[feature] = fill_function(df_clean[feature])
        return df_clean


class MissingValuesHandler(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: MissingValuesHandlingStrategy):
        self._strategy = strategy
        self.feature_names_in_ = []

    @property
    def strategy(self):
        return self._strategy

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X: pd.DataFrame):
        transformed_df = self._strategy.handle_missing_values(X)
        return transformed_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return self.feature_names_in_