import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionModel:
    def __init__(self, X, y):
        """
        Initialize the class.
        X: DataFrame containing the descriptors.
        y: Series or DataFrame with target values to predict.
        """
        self.X = X
        self.y = y
        self.models = {}
        self.best_model = None
        self.metrics = {}

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and test sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train_linear_regression(self):
        """
        Train a simple linear regression model and save its metrics.
        """
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.models["LinearRegression"] = model
        self.evaluate_model("LinearRegression", model)

    def train_with_grid_search(self, model_name, model, param_grid):
        """
        Perform GridSearchCV for a given model.
        model_name: Name of the model.
        model: The model object.
        param_grid: Dictionary of hyperparameters to search.
        """
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        self.models[model_name] = best_model
        self.evaluate_model(model_name, best_model)

    def evaluate_model(self, model_name, model):
        """
        Compute metrics for a given model and save them.
        model_name: Name of the model.
        model: The model object.
        """
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        self.metrics[model_name] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }

    def find_best_model(self):
        """
        Identify the best model based on R2 score.
        """
        best_model_name = max(self.metrics, key=lambda name: self.metrics[name]["R2"])
        self.best_model = self.models[best_model_name]
        return best_model_name, self.metrics[best_model_name]

    def run(self):
        """
        Execute all steps: split data, train models, and select the best one.
        """
        self.split_data()
        
        # Train linear regression
        self.train_linear_regression()
        
        # Perform grid search for Ridge Regression
        self.train_with_grid_search(
            "Ridge", 
            Ridge(), 
            param_grid={"alpha": [0.1, 1, 10, 100]}
        )
        
        # Perform grid search for Lasso Regression
        self.train_with_grid_search(
            "Lasso", 
            Lasso(), 
            param_grid={"alpha": [0.01, 0.1, 1, 10]}
        )
        
        # Find the best model
        best_model_name, best_metrics = self.find_best_model()
        print(f"Best model: {best_model_name}")
        print(f"Metrics: {best_metrics}")
