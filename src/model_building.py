import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import preprocessor

logger = logging.getLogger(__name__)


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Abstract method for building and training a model.

        :param:
            X_train (pd.DataFrame): The training data features
            y_train (pd.Series): The training data labels/targets
        :return:
            RegressorMixin: The trained model instance
        """
        pass


class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """

        :param:
            X_train (pd.DataFrame): The training data features
            y_train (pd.Series): The training data labels/targets
        :return:
            Pipeline: A pipeline with a trained regression model instance
        """
        # if not isinstance(X_train, pd.DataFrame):
        #     raise TypeError("X_train must be a pandas.DataFrame")
        # elif not isinstance(y_train, pd.Series):
        #     raise TypeError("y_train must be a pandas.Series")

        logger.info("Initializing regression model")

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        logger.info("Training Linear Regression model")
        pipeline.fit(X_train, y_train)

        return pipeline


class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the Model builder with a specified model building strategy.

        :param:
            strategy (ModelBuildingStrategy): The strategy to use for building the model
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder.

        :param:
            strategy (ModelBuildingStrategy): The new strategy to use for building the model
        """
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Executes the model building and training using the current strategy.

        :param:
            X_train (pd.DataFrame): The training data features
            y_train (pd.Series): The training data labels/targets
        :return:
            Pipeline: A pipeline with a trained regression model instance
        """
        logger.info("Building and training model using the set strategy")
        return self._strategy.build_and_train_model(X_train, y_train)


if __name__ == '__main__':
    import os

    from sklearn.metrics import root_mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    from src.ingest_data import DataIngestorFactory

    file_path = "/Users/ktxdev/Developer/house-prices/data/train.csv"
    file_extension = os.path.splitext(file_path)[1]

    data_ingestor = DataIngestorFactory().get_data_ingestor(file_extension)
    data = data_ingestor.ingest(file_path)

    X = data.drop(columns=['Id', 'SalePrice'])
    y = data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))

    model_builder = ModelBuilder(LinearRegressionStrategy())
    model = model_builder.build_model(X_train, y_train_scaled)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    print("Sample predictions:", y_pred[:10])
    print(root_mean_squared_error(y_test, y_pred))
    print(r2_score(y_test, y_pred))
