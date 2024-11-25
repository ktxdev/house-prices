import pandas as pd
from .category_mappings import mappings


def map_and_convert_categories(data: pd.DataFrame) -> pd.DataFrame:
    # Make a copy
    data = data.copy()

    # map each column
    for key, value in mappings.items():
        data[key] = data[key].map(value).astype('category')

    return data
