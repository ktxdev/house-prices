import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        """
        Abstract method for building and training a model.

        :param:
            X_train (pd.DataFrame): The training data features
            y_train (pd.Series): The training data labels/targets
        :return:
            RegressorMixin: The trained model instance
        """
        pass


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

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
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
