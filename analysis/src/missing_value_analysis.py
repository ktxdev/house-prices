import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class MissingValueAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame) -> None:
        """
        Performs missing value analysis on a dataframe.
        param
            df (pd.DataFrame): Dataframe to check for missing values
        """
        self.indentify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def indentify_missing_values(self, df: pd.DataFrame) -> None:
        """
        Identify missing values in dataframe.
        param
            df(pd.DataFrame): Dataframe to check for missing values
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        """
        Visualize missing values in dataframe.
        param
            df(pd.DataFrame): Dataframe to visualize missing values for
        """
        pass


class SimpleMissingValueAnalysis(MissingValueAnalysisTemplate):
    def indentify_missing_values(self, df: pd.DataFrame) -> None:
        """
        Prints the count of missing values for each column in dataframe.
        param
            df(pd.DataFrame): Dataframe to check for missing values
        """
        print("\nMissing values Count by Column:")
        missing_values_count = df.isnull().sum()
        print(missing_values_count[missing_values_count > 0])

    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        """
        Creates a heatmap to visualize missing values for each column in dataframe.
        param
            df(pd.DataFrame): Dataframe to visualize missing values for
        """
        plt.figure(figsize=(20, 12))
        sns.heatmap(df.isnull(), annot=True, cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
