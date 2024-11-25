import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.logger import get_logger, log_message

logger = get_logger('DataVisualiser')

SHOW_LOGS = False
DEFAULT_FIGSIZE = (15, 6)


class DataVisualiser:
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

    def plot(self, column: str) -> None:
        if not self._column_exists(column):
            return

        col_dtype = self.data[column].dtype

        if pd.api.types.is_numeric_dtype(col_dtype):
            self._plot_numerical(column)
