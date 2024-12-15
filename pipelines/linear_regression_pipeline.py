import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import preprocessor
from src.model_building import ModelBuildingStrategy
from src.model_evaluation import rmse_scorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        """

        :param:
            X_train (pd.DataFrame): The training data features
            y_train (pd.Series): The training data labels/targets
        :return:
            Pipeline: A pipeline with a trained regression model instance
        """

        logger.info("Initializing regression model")

        param_grid = {
            'model__fit_intercept': [True, False],
            'model__n_jobs': [1, -1]
        }

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                                   scoring=make_scorer(rmse_scorer, greater_is_better=False), cv=5)

        logger.info("Training Linear Regression model")
        grid_search.fit(X_train, y_train)

        return grid_search
