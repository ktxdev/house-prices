from typing import Dict, Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from pipelines.support_vector_regression_pipeline import SupportVectorRegressionStrategy
from src.experiment_logger import log_experiment
from src.model_building import ModelBuilder
from src.model_evaluation import ModelEvaluator


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
    import joblib

    from sklearn.preprocessing import StandardScaler

    from src.ingest_data import DataIngestorFactory
    from src.outlier_detection import OutlierHandlerCap
    from src.model_evaluation import RegressionPipelineEvaluationStrategy

    file_path = "/Users/ktxdev/Developer/house-prices/data/train.csv"
    file_extension = os.path.splitext(file_path)[1]

    data_ingestor = DataIngestorFactory().get_data_ingestor(file_extension)
    data = data_ingestor.ingest(file_path)

    X = data.drop(columns=['Id', 'SalePrice'])
    y = data['SalePrice']

    outlier_handler = OutlierHandlerCap(method='iqr', threshold=1.5)
    y = outlier_handler.transform(y)

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    model_builder = ModelBuilder(SupportVectorRegressionStrategy())

    model_evaluator = ModelEvaluator(RegressionPipelineEvaluationStrategy())

    model_trainer = ModelTrainer(model_builder, model_evaluator)

    grid_search, metrics = model_trainer.train_and_evaluate_model(X, y_scaled)

    model_name = "SupportVectorRegression_v1.0"
    log_experiment(model_name, "Model with outliers capped using iqr", metrics)

    models_dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../models")
    os.makedirs(models_dir_path, exist_ok=True)
    # Save the best pipeline to a file
    joblib.dump(grid_search.best_estimator_, os.path.join(models_dir_path, f"{model_name}.pkl"))
    # Save the scaler to a file
    joblib.dump(scaler, os.path.join(models_dir_path, f"scaler.pkl"))
