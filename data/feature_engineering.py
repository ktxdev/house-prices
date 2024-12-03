import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def handle_rare_categories(data: pd.DataFrame) -> pd.DataFrame:
    # Make copy of dataframe
    data = data.copy()
    # Identify rare categories in the 'MSSubClass' feature
    rare_categories = data['MSSubClass'].value_counts()[data['MSSubClass'].value_counts() < 20].index
    # Replace rare categories with 'Other'
    data['MSSubClass'] = data['MSSubClass'].apply(lambda x: 'Other' if x in rare_categories else x)
    return data

def encode_categories(data: pd.DataFrame) -> pd.DataFrame:
    # Make copy of dataframe
    data = data.copy()
    # Frequency encode MSSubClass
    freq_map = data['MSSubClass'].value_counts().to_dict()
    data['MSSubClass_Encoded'] = data['MSSubClass'].map(freq_map)
    # One hot encode MSZoning
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(data[['MSZoning']]).toarray()
    # Create dataframe of encoded values
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['MSZoning']))
    # Concatenate encoded_df to data
    data = pd.concat([data, encoded_df], axis=1).drop(columns=['MSZoning'])
    return data

def normalize_numerics(data: pd.DataFrame) -> pd.DataFrame:
    # Make copy of dataframe
    data = data.copy()
    scaler = MinMaxScaler()
    data['LotFrontage'] = scaler.fit_transform(data[['LotFrontage']])
    return data