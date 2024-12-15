import os
import pandas as pd

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data into a pandas dataframe"""
        pass


class CSVDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from .csv file into a pandas dataframe"""
        df = pd.read_csv(file_path)
        return df


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Get data ingestor based on file extension"""
        if file_extension == '.csv':
            return CSVDataIngestor()
        else:
            raise ValueError(f'No data ingestor available for file extension {file_extension}')


