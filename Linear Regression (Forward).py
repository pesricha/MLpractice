import numpy as np
from numpy.typing import NDArray


# Helpful functions:
# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
# https://numpy.org/doc/stable/reference/generated/numpy.square.html

class Solution:
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is an Nx3 NumPy array
        # weights is a 3x1 NumPy array
        # HINT: np.matmul() will be useful
        prediction = np.matmul(X, weights)
        # return np.round(your_answer, 5)
        return np.round(prediction, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # model_prediction is an Nx1 NumPy array
        # ground_truth is an Nx1 NumPy array
        # HINT: np.mean(), np.square() will be useful
        errors = model_prediction-ground_truth
        squared_errors = np.square(errors)
        mean_standard_error = np.mean(squared_errors)
        # return round(your_answer, 5)
        return np.round(mean_standard_error, 5)

