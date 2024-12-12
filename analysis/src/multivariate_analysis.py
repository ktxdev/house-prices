from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MultivariateAnalysisTemplate(ABC):
    def analyse(self, df: pd.DataFrame) -> None:
        """
        Performs multivariate analysis on a dataframe by generating a correlation heatmap and pair plots.
        :param:
            df (pd.DataFrame): Pandas dataframe containing the data to be analysed.
        :return:
            None: Visualizes the heatmap and pair plots.
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplots(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """
        Generates a correlation heatmap of the dataframe.
        :param:
            df (pd.DataFrame): Pandas dataframe containing the data to be analysed.
        :return:
            None: Visualizes the heatmap.
        """
        pass

    @abstractmethod
    def generate_pairplots(self, df: pd.DataFrame) -> None:
        """
        Generates a pair plots of the dataframe.
        :param:
            df (pd.DataFrame): Pandas dataframe containing the data to be analysed.
        :return:
            None: Visualizes the pair plots.
        """
        pass


class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """
        Generates a correlation heatmap of the dataframe.
        :param:
            df (pd.DataFrame): Pandas dataframe containing the data to be analysed.
        :return:
            None: Visualizes the heatmap.
        """
        plt.figure(figsize=(40, 20))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()

    def generate_pairplots(self, df: pd.DataFrame) -> None:
        pass
