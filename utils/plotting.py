import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.logger import get_logger, log_message

logger = get_logger('DataVisualiser')

SHOW_LOGS = False
DEFAULT_FIGSIZE = (15, 6)


class DataVisualizer:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def _column_exists(self, column: str) -> bool:
        if column not in self.data.columns:
            log_message(f"Column '{column}' does not exist in the DataFrame.", logger, SHOW_LOGS)
            return False
        return True

    @staticmethod
    def _customize_plot(ax: plt.Axes, title: str, xlabel: str, ylabel: str, rotation: int = 0) -> None:
        """Helper function to add customizations to plots."""
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=rotation)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    def _plot_numerical(self, column: str) -> None:
        """Plot a histogram and boxplot of numerical datasets"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE)
        sns.histplot(data=self.data, x=column, ax=ax1, kde=True)
        sns.boxplot(data=self.data, x=column, ax=ax2)
        self._customize_plot(ax1, f"Histogram of {column}", column, "Frequency")
        self._customize_plot(ax2, f"Boxplot of {column}", column, "Value")
        plt.tight_layout()
        plt.show()

    def _plot_categorical(self, column: str) -> None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        sns.countplot(data=self.data, x=column, ax=ax)
        self._customize_plot(ax, f"Count of {column}", column, "Frequency", rotation=90)
        plt.show()

    def _plot_numerical_vs_categorical(self, numerical_col: str, categorical_col: str) -> None:
        plt.figure(figsize=DEFAULT_FIGSIZE)
        sns.boxplot(data=self.data, x=categorical_col, y=numerical_col)
        plt.title(f"{numerical_col} vs. {categorical_col}")
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def _plot_numerical_vs_numerical(self, numeric_col1: str, numeric_col2: str) -> None:
        plt.figure(figsize=DEFAULT_FIGSIZE)
        sns.scatterplot(data=self.data, x=numeric_col1, y=numeric_col2)
        plt.title(f"{numeric_col1} vs. {numeric_col2}")
        plt.xlabel(numeric_col1)
        plt.ylabel(numeric_col2)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

    def _is_categorical(self, column: str) -> bool:
        col_dtype = self.data[column].dtype
        return isinstance(col_dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(col_dtype)

    def _is_numerical(self, column: str) -> bool:
        col_dtype = self.data[column].dtype
        return pd.api.types.is_numeric_dtype(col_dtype)

    def plot(self, column1: str, column2: str = None) -> None:
        if not self._column_exists(column1):
            return

        if column2:
            if self._is_categorical(column2) and self._is_numerical(column1):
                self._plot_numerical_vs_categorical(column1, column2)
            elif self._is_categorical(column1) and self._is_numerical(column2):
                self._plot_numerical_vs_categorical(column2, column1)
            elif self._is_numerical(column1) and self._is_numerical(column2):
                self._plot_numerical_vs_numerical(column1, column2)
        else:
            if self._is_numerical(column1):
                self._plot_numerical(column1)
            elif self._is_categorical(column1):
                self._plot_categorical(column1)
