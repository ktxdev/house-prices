import pandas as pd


def handle_rare_categories(data: pd.DataFrame) -> pd.DataFrame:
    # Clone dataframe
    data = data.copy()
    # Identify rare categories in the 'MSSubClass' feature
    rare_categories = data['MSSubClass'].value_counts()[data['MSSubClass'].value_counts() < 20].index
    # Replace rare categories with 'Other'
    data['MSSubClass'] = data['MSSubClass'].apply(lambda x: 'Other' if x in rare_categories else x)
    return data