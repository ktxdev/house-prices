from typing import Dict, Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from src.model_building import ModelBuilder
from src.model_evaluation import ModelEvaluator
from src.experiment_logger import log_experiment


class ModelTrainer:
    def __init__(self, model_builder: ModelBuilder, model_evaluator: ModelEvaluator):
        self._model_builder = model_builder
        self._model_evaluator = model_evaluator

    def train_and_evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[GridSearchCV, Dict[str, Any]]:
        X = data.drop(columns=['Id', 'SalePrice'])
        y = data['SalePrice']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = model_builder.build_model(X_train, y_train)
        metrics = model_evaluator.evaluate(model, X_test, y_test)

        return model, metrics


if __name__ == '__main__':
    import os

    from sklearn.preprocessing import StandardScaler

    from src.ingest_data import DataIngestorFactory
    from pipelines.linear_regression_pipeline import LinearRegressionStrategy
    from src.model_evaluation import RegressionPipelineEvaluationStrategy

    file_path = "/Users/ktxdev/Developer/house-prices/data/train.csv"
    file_extension = os.path.splitext(file_path)[1]

    data_ingestor = DataIngestorFactory().get_data_ingestor(file_extension)
    data = data_ingestor.ingest(file_path)

    X = data.drop(columns=['Id', 'SalePrice'])
    y = data['SalePrice']

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    model_builder = ModelBuilder(LinearRegressionStrategy())

    model_evaluator = ModelEvaluator(RegressionPipelineEvaluationStrategy())

    model_trainer = ModelTrainer(model_builder, model_evaluator)

    model, metrics = model_trainer.train_and_evaluate_model(X, y_scaled)

    log_experiment("LinearRegression_v2.0", "Model with a large count of missing values removed", metrics)
