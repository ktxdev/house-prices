import logging
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.data_preprocessing import preprocessor
from src.model_building import ModelBuildingStrategy
from src.model_evaluation import rmse_scorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GradientBoostingRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        logger.info("Initializing Gradient Boosting Regression model")

        # Define the parameter grid
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(random_state=42))
        ])

        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                                   scoring=make_scorer(rmse_scorer, greater_is_better=False), n_jobs=-1, cv=5,
                                   verbose=3)
        logger.info("Training Gradient Boosting Regression model")
        grid_search.fit(X_train, y_train)

        return grid_search
