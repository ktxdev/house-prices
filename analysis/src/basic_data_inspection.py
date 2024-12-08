import pandas as pd

from abc import ABC, abstractmethod


class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Perform data inspection on the given dataframe.

        :param
            df (pd.DataFrame): Dataframe to inspect
        :return:
            None: Prints results of data inspection directly
        """
        pass


class DataTypeInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Perform data type inspection on the given dataframe.
        :param
            df (pd.DataFrame): Dataframe to inspect
        :return:
            None: Prints results of data type inspection directly
        """
        print("\nData types and Non-null counts")
        print(df.info())


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Perform summary statistics inspection on the given dataframe.
        :param
            df (pd.DataFrame): Dataframe to inspect
        :return:
            None: Prints results of summary statistics inspection directly
        """
        print("\nSummary statistics (Numerical Features)")
        print(df.describe())
        print("\nSummary statistics (Categorical Features)")
        print(df.describe(include=['object']))


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initialize the data inspector class with a specified inspection strategy.

        param
            strategy (DataInspectionStrategy): The strategy to be used for inspection
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame) -> None:
        """
        Perform data inspection using the specified inspection strategy.

        param
            df (pd.DataFrame): Dataframe to inspect
        """
        self._strategy.inspect(df)

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Set the data inspector strategy.

        param
            strategy (DataInspectionStrategy): The new strategy to be used for inspection
        """
        self._strategy = strategy
        return self
