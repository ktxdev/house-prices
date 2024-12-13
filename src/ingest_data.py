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


class DataIngestor(BaseEstimator, TransformerMixin):
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._file_extension = os.path.splitext(file_path)[1]

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        data_ingestor_factory = DataIngestorFactory.get_data_ingestor(self._file_extension)
        return data_ingestor_factory.ingest(self._file_path)
