import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """
        Performs univariate analysis on a specific feature in the dataframe.

        :param
            df (pd.DataFrame): The dataframe to use for univariate analysis.
            feature (str): The feature to analyze.
        :return
            None: Visualizes the distribution of the feature.
        """
        pass


class NumericUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """
        Plots the distribution of a numerical feature using a histogram, KDE and boxplot
        :param
            df (pd.DataFrame): The dataframe to use for univariate analysis.
            feature (str): The feature to analyze.
        :return:
            None: Visualizes the distribution of the feature.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Histgram with KDE
        sns.histplot(data=df, x=feature, ax=ax1, kde=True)
        ax1.set_title(f"Histogram of {feature}")
        ax1.set_xlabel(feature)
        ax1.set_ylabel("Frequency")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Box plot
        sns.boxplot(data=df, x=feature, ax=ax2)
        ax2.set_title(f"Boxplot of {feature}")
        ax2.set_xlabel(feature)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """
        Plots the distribution of a categorical feature using a countplot
        :param
            df(pd.DataFrame): The dataframe to use for univariate analysis.
            feature (str): The feature to analyze.
        :return:
            None: Visualizes the distribution of the feature.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        # Count Plot
        sns.countplot(data=df, x=feature, ax=ax)
        ax.set_title(f"Count of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.show()

class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes a univariate analyzer with a specific strategy.
        :param
            strategy (UnivariateAnalysisStrategy): The strategy to use.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets the strategy to use.
        :param
            strategy (UnivariateAnalysisStrategy): The strategy to use.
        """
        self.strategy = strategy
        return self

    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """
        Performs univariate analysis on a specific feature in the dataframe.
        :param
            df (pd.DataFrame): The dataframe to use for univariate analysis.
            feature (str): The feature to analyze.
        """
        self.strategy.analyze(df, feature)
