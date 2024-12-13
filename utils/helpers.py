import os

import pandas as pd

from utils.logger import get_logger, log_message

SHOW_LOGS: bool = False

logger = get_logger(__name__)

DATASET_DIR = data_path = os.path.abspath(os.path.join('..', 'datasets'))

def load_data(data_filename: str) -> pd.DataFrame:
    data_path = os.path.join(DATASET_DIR, data_filename)
    log_message(f"Loading datasets from: {data_path}", logger, SHOW_LOGS)
    return pd.read_csv(data_path)

def save_data(data_filename: str, data: pd.DataFrame) -> None:
    data_path = os.path.join(DATASET_DIR, data_filename)
    log_message(f"Saving datasets to: {data_path}", logger, SHOW_LOGS)
    data.to_csv(data_path, index=False)


def check_and_print_missing_value_counts(data: pd.DataFrame, column_name: str) -> None:
    """Counts and displays missing value counts for given column_name"""
    missing_values_rows = data[column_name].isnull()
    # Count missing values
    missing_values_count = missing_values_rows.sum()
    # Calculate missing values percentage
    missing_values_percentage = round(missing_values_rows.mean() * 100, 2)
    print(f"Missing values count: {missing_values_count}\nMissing values percentage: {missing_values_percentage}%")

import os
from dotenv import load_dotenv

def get_env_variable(var_name: str) -> str:
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(dotenv_path)

    return os.getenv(var_name)