import pandas as pd

from abc import ABC, abstractmethod


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


if __name__ == '__main__':
    import os

    # Specify file path
    file_path = '/Users/ktxdev/Developer/house-prices/datasets/train.csv'
    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]
    # Get the appropriate data ingestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    # Ingest the data and load it into a dataframe
    df = data_ingestor.ingest(file_path)
    # Show top records
    print(df.head())
