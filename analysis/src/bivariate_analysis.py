import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Performs bivariate analysis on specific features in the dataframe.
        :param:
            df (pd.DataFrame): Dataframe with features to analyze.
            feature1 (str): Name of the 1st feature to analyze.
            feature2 (str): Name of the 2nd feature to analyze.
        :return:
            None: Visualizes bivariate analysis.
        """
        pass

    def _setup_plot(self, title: str, xlabel: str, ylabel: str, rotate_x: bool = False) -> None:
        """
        Sets up the plot with title, labels, and grid.
        :param:
            title: Title of the plot.
            xlabel: Label for the X-axis.
            ylabel: Label for the Y-axis.
            rotate_x: Whether to rotate the X-axis labels.
        """
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if rotate_x:
            plt.tick_params(axis='x', rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)


class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Performs numerical bivariate analysis on specific features in the dataframe.
        :param:
            df (pd.DataFrame): Dataframe with features to analyze.
            feature1 (str): Name of the 1st feature to analyze.
            feature2 (str): Name of the 2nd feature to analyze.
        :return:
            None: Visualizes numerical bivariate analysis using a scatter plot.
        """
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x=feature1, y=feature2)
        self._setup_plot(title=f"{feature1} vs. {feature2}", xlabel=feature1, ylabel=feature2)
        plt.show()


class NumericalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Performs numerical vs categorical bivariate analysis on specific features in the dataframe.
        :param:
            df (pd.DataFrame): Dataframe with features to analyze.
            feature1 (str): Name of the 1st feature to analyze.
            feature2 (str): Name of the 2nd feature to analyze.
        :return:
            None: Visualizes numerical vs categorical bivariate analysis using a box plot.
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x=feature1, y=feature2)
        self._setup_plot(title=f"{feature1} vs. {feature2}", xlabel=feature1, ylabel=feature2, rotate_x=True)
        plt.show()

class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the bivariate analyzer with a specific strategy.
        :param:
            strategy (BivariateAnalysisStrategy): Strategy to be used.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets the bivariate analysis strategy.
        :param:
            strategy (BivariateAnalysisStrategy): Strategy to be used.
        """
        self._strategy = strategy
        return self

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Performs bivariate analysis on the specified features in the dataframe.
        :param:
            df (pd.DataFrame): Dataframe with features to analyze.
            feature1 (str): Name of the 1st feature to analyze.
            feature2 (str): Name of the 2nd feature to analyze.
        """
        self._strategy.analyze(df, feature1, feature2)