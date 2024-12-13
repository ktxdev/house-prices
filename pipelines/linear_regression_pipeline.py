from joblib.parallel import method
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.ingest_data import DataIngestorFactory
from src.missing_values_handling import MissingValuesHandler, FillMissingValuesStrategy
from utils.helpers import get_env_variable

DATA_FILE_PATH = get_env_variable('DATA_FILE_PATH')

basement_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

preprocessor = ColumnTransformer(
    transformers=[
        # ('drop_features', MissingValuesHandler(DropMissingValuesStrategy(features=features_to_drop)), features_to_drop),
        ('lot_frontage', MissingValuesHandler(FillMissingValuesStrategy(['LotFrontage'], method="median", grouping_feature='Neighborhood')), ['LotFrontage', 'Neighborhood']),
        ('alley', MissingValuesHandler(FillMissingValuesStrategy(['Alley'], method="constant", fill_value='No Alley')), ['Alley']),
        ('mas_vnr_type', MissingValuesHandler(FillMissingValuesStrategy(['MasVnrType'], method="constant", fill_value='None')), ['MasVnrType']),
        ('mas_vnr_area', MissingValuesHandler(FillMissingValuesStrategy(['MasVnrArea'], method="constant", fill_value=0)), ['MasVnrArea']),
        ('bsmt', MissingValuesHandler(FillMissingValuesStrategy(basement_features, method="constant", fill_value="No Basement")), basement_features),
        ('electrical', MissingValuesHandler(FillMissingValuesStrategy(['Electrical'], method="most_frequent")), ['Electrical']),
    ],
    remainder='passthrough',
)

linear_regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
])

if __name__ == '__main__':
    import os

    file_path = '/Users/ktxdev/Developer/house-prices/datasets/train.csv'
    file_extension = os.path.splitext(file_path)[1]

    data_ingestor_factory = DataIngestorFactory.get_data_ingestor(file_extension)
    data = data_ingestor_factory.ingest(file_path)

    X = data.drop(columns=['Id', 'SalePrice'])
    y = data['SalePrice']

    # Pass the ingested data through the pipeline
    transformed_data = linear_regression_pipeline.fit_transform(X, y)

    print("Transformed Data:", transformed_data.shape)

