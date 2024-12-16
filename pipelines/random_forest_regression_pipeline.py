import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.data_preprocessing import preprocessor
from src.model_building import ModelBuildingStrategy
from src.model_evaluation import rmse_scorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RandomForestRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        logger.info("Initializing random forest model")

        # Define the parameter grid
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=42))
        ])

        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                                   scoring=make_scorer(rmse_scorer, greater_is_better=False), cv=5, verbose=3)
        logger.info("Training Random Forest Regression model")
        grid_search.fit(X_train, y_train)

        return grid_search
