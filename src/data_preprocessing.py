from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.missing_values_handling import MissingValuesHandler, FillMissingValuesStrategy

numeric_columns = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1',
                   'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                   'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                   'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                   'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

categorical_features = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
                        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                        'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation',
                        'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual',
                        'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']

lot_frontage_column_transformer = Pipeline(steps=[
    ('imputer', MissingValuesHandler(FillMissingValuesStrategy(['LotFrontage'], method="median"))),
    ('scaler', StandardScaler())
])

alley_column_transformer = Pipeline(steps=[
    ('imputer', MissingValuesHandler(FillMissingValuesStrategy(['Alley'], method="constant", fill_value='No Alley'))),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

mas_vnr_type_column_transformer = Pipeline(steps=[
    ('imputer', MissingValuesHandler(FillMissingValuesStrategy(['MasVnrType'], method="constant", fill_value='None'))),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

mas_vnr_area_column_transformer = Pipeline(steps=[
    ('imputer', MissingValuesHandler(FillMissingValuesStrategy(['MasVnrArea'], method="constant", fill_value=0))),
    ('scaler', StandardScaler())
])

basement_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

basement_features_column_transformer = Pipeline(steps=[
    ('imputer',
     MissingValuesHandler(FillMissingValuesStrategy(basement_features, method="constant", fill_value="No Basement"))),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

electrical_column_transformer = Pipeline(steps=[
    ('imputer', MissingValuesHandler(FillMissingValuesStrategy(['Electrical'], method="most_frequent"))),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

fireplace_qu_column_transformer = Pipeline(steps=[
    ('imputer',
     MissingValuesHandler(FillMissingValuesStrategy(['FireplaceQu'], method="constant", fill_value='No Fireplace'))),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

garage_categorical_features = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

garage_features_column_transformer = Pipeline(steps=[
    ('imputer', MissingValuesHandler(
        FillMissingValuesStrategy(garage_categorical_features, method="constant", fill_value='No Garage'))),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

garage_yr_blt_column_transformer = Pipeline(steps=[
    ('imputer', MissingValuesHandler(FillMissingValuesStrategy(['GarageYrBlt'], method="constant", fill_value=0))),
    ('scaler', StandardScaler())
])

pool_qc_column_transformer = Pipeline(steps=[
    ('imputer', MissingValuesHandler(FillMissingValuesStrategy(['PoolQC'], method="constant", fill_value='No Pool'))),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

fence_column_transformer = Pipeline(steps=[
    ('imputer', MissingValuesHandler(FillMissingValuesStrategy(['Fence'], method="constant", fill_value='No Fence'))),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

misc_feature_column_transformer = Pipeline(steps=[
    ('imputer', MissingValuesHandler(FillMissingValuesStrategy(['MiscFeature'], method="constant", fill_value='None'))),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        # ('lot_frontage', lot_frontage_column_transformer, ['LotFrontage']),
        # ('alley', alley_column_transformer, ['Alley']),
        # ('mas_vnr_type', mas_vnr_type_column_transformer, ['MasVnrType']),
        # ('mas_vnr_area', mas_vnr_area_column_transformer, ['MasVnrArea']),
        # ('basement', basement_features_column_transformer, basement_features),
        # ('electrical', electrical_column_transformer, ['Electrical']),
        # ('fireplace', fireplace_qu_column_transformer, ['FireplaceQu']),
        # ('garage', garage_features_column_transformer, garage_categorical_features),
        # ('garage_yr_blt', garage_yr_blt_column_transformer, ['GarageYrBlt']),
        # ('pool_qc', pool_qc_column_transformer, ['PoolQC']),
        # ('fence', fence_column_transformer, ['Fence']),
        # ('misc_feature', misc_feature_column_transformer, ['MiscFeature']),
        ('num', SimpleImputer(strategy='median'), numeric_columns),
        ('cat', SimpleImputer(strategy="most_frequent"), categorical_features),
        ('scaler', StandardScaler(), numeric_columns),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop',
)


if __name__ == '__main__':
    import os

    from sklearn.model_selection import train_test_split

    from src.ingest_data import DataIngestorFactory

    file_path = "/Users/ktxdev/Developer/house-prices/data/train.csv"
    file_extension = os.path.splitext(file_path)[1]

    data_ingestor = DataIngestorFactory().get_data_ingestor(file_extension)
    data = data_ingestor.ingest(file_path)

    X = data.drop(columns=['Id', 'SalePrice'])
    y = data['SalePrice']

    # transformed_data = preprocessor.fit_transform(X)
    #
    # print(transformed_data.shape)
    intermediate_result = lot_frontage_column_transformer.fit_transform(X)
    print("Intermediate result for 'LotFrontage':", intermediate_result[:5])
