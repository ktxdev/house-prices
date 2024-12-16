import numpy as np
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierHandlerCap(BaseEstimator, TransformerMixin):
    def __init__(self, method='zscore', threshold=3.0, lower_cap=None, upper_cap=None):
        """
        Outlier handling transformer that caps the outliers to specified ranges.

        Parameters:
        - method: 'zscore' or 'iqr', the method for outlier detection
        - threshold: The threshold value for detecting outliers (z-score or IQR).
        - lower_cap: Lower value to cap the outliers. If None, the lower bound from the method will be used.
        - upper_cap: Upper value to cap the outliers. If None, the upper bound from the method will be used.
        """
        self.method = method
        self.threshold = threshold
        self.lower_cap = lower_cap
        self.upper_cap = upper_cap

    def fit(self, X, y=None):
        """
        Fit the transformer. In this case, we do not need to fit any parameters,
        but it is required by scikit-learn.
        """
        return self

    def transform(self, X):
        """
        Transform the data by capping outliers.

        If using Z-score, it caps values with z-score > threshold to the upper or lower cap.
        If using IQR, it caps values outside the IQR range to the upper or lower cap.
        """
        X_transformed = X.copy()

        if self.method == 'zscore':
            # Calculate z-scores
            z_scores = np.abs(zscore(X_transformed, nan_policy='omit'))
            # Define upper and lower bounds for capping based on z-score threshold
            lower_bound = -self.threshold
            upper_bound = self.threshold
            # Cap values based on z-score threshold
            X_transformed[z_scores < lower_bound] = lower_bound
            X_transformed[z_scores > upper_bound] = upper_bound

        elif self.method == 'iqr':
            # Calculate the IQR
            Q1 = np.percentile(X_transformed, 25, axis=0)
            Q3 = np.percentile(X_transformed, 75, axis=0)
            IQR = Q3 - Q1
            # Define bounds for capping
            lower_bound = Q1 - 1.5 * IQR if self.lower_cap is None else self.lower_cap
            upper_bound = Q3 + 1.5 * IQR if self.upper_cap is None else self.upper_cap
            # Cap outliers
            X_transformed = np.clip(X_transformed, lower_bound, upper_bound)

        return X_transformed
