import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.missing_values_handling import MissingValuesHandler, FillMissingValuesStrategy, DropMissingValuesStrategy

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the column groups
median_columns = ['LotFrontage']
constant_none = ['MasVnrType']
constant_zero = ['MasVnrArea', 'GarageYrBlt']
constant_no_basement = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
most_frequent_columns = ['Electrical']
constant_no_fireplace = ['FireplaceQu']
constant_no_garage = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

# Create handlers for each group of columns
missing_value_transformers = [
    ('median_imputer', Pipeline(steps=[
        ('imputer', MissingValuesHandler(FillMissingValuesStrategy(median_columns, method="median"))),
        ('scaler', StandardScaler())
    ]),
     median_columns),
    ('constant_none',
     Pipeline(steps=[
         ('imputer',
          MissingValuesHandler(FillMissingValuesStrategy(constant_none, method="constant", fill_value='None'))),
         ('encoder', OneHotEncoder(handle_unknown='ignore'))
     ]),
     constant_none),
    ('constant_zero', Pipeline(steps=[
        ('imputer', MissingValuesHandler(FillMissingValuesStrategy(constant_zero, method="constant", fill_value=0))),
        ('scaler', StandardScaler())
    ]),
     constant_zero),
    ('constant_no_basement', Pipeline(steps=[
        ('imputer', MissingValuesHandler(
            FillMissingValuesStrategy(constant_no_basement, method="constant", fill_value="No Basement"))),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]),
     constant_no_basement),
    ('most_frequent', Pipeline(steps=[
        ('imputer', MissingValuesHandler(FillMissingValuesStrategy(most_frequent_columns, method="most_frequent"))),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]),
     most_frequent_columns),
    ('constant_no_fireplace', Pipeline(steps=[
        ('imputer', MissingValuesHandler(
            FillMissingValuesStrategy(constant_no_fireplace, method="constant", fill_value='No Fireplace'))),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]),
     constant_no_fireplace),
    ('constant_no_garage', Pipeline(steps=[
        ('imputer', MissingValuesHandler(
            FillMissingValuesStrategy(constant_no_garage, method="constant", fill_value='No Garage'))),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]),
     constant_no_garage),
]

numerical_features = ['MSSubClass', 'LotArea', 'OverallQual',
                      'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1',
                      'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                      'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                      'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                      'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                      'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                      'MiscVal', 'MoSold', 'YrSold']
categorical_features = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
                        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                        'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation',
                        'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual',
                        'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']

# Combine missing value handling, scaling, and encoding
preprocessor = ColumnTransformer(
    transformers=[
        *missing_value_transformers,  # Missing value handling
        ('scaler', StandardScaler(), numerical_features),  # Scaling for numerical columns
        ('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features),  # One-hot encoding
    ],
    remainder='drop'  # Retain remaining columns
)

if __name__ == '__main__':
    import os

    from src.ingest_data import DataIngestorFactory

    file_path = "/Users/ktxdev/Developer/house-prices/data/train.csv"
    file_extension = os.path.splitext(file_path)[1]

    data_ingestor = DataIngestorFactory().get_data_ingestor(file_extension)
    data = data_ingestor.ingest(file_path)

    # Integrate into a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
    ])

    data_transformed = pipeline.fit_transform(data)
    print(data_transformed[:10])
