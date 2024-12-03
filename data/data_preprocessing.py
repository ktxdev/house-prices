import pandas as pd
from sklearn.impute import KNNImputer

from .category_mappings import mappings


def map_and_convert_categories(data: pd.DataFrame) -> pd.DataFrame:
    # Make a copy
    data = data.copy()

    # map each column
    for key, value in mappings.items():
        data[key] = data[key].map(value).astype('category')

    return data

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    # Copy dataframe
    data = data.copy()
    # Impute MSZoning missing values
    data['MSZoning'] = data.groupby('MSSubClass', observed=True)['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    # Impute using k-NN
    imputer = KNNImputer(n_neighbors=5)
    data[['LotFrontage']] = imputer.fit_transform(data[['LotFrontage']])
    return data

def handle_outliers(data: pd.DataFrame) -> pd.DataFrame:
    # Copy dataframe
    data = data.copy()
    # Cap extreme values
    columns_to_cap = ['LotFrontage', 'LotArea']
    # Compute the lower (5th percentile) and upper (95th percentile) bounds
    lower_bounds = data[columns_to_cap].quantile(0.05)
    upper_bounds = data[columns_to_cap].quantile(0.95)
    # Use clip to cap the values within the specified range
    data[columns_to_cap] = data[columns_to_cap].clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    return data