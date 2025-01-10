import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class RegressionModel:
    def __init__(self, X, y, random_state=42):
        self.X = X
        self.y = y
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.metrics = {}
        self.scaler = StandardScaler()
        self.random_state = random_state

    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        self.normalize_features()

    def normalize_features(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_linear_regression(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.models["LinearRegression"] = model
        self.evaluate_model("LinearRegression", model)

    def train_with_grid_search(self, model_name, model, param_grid):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        self.models[model_name] = best_model
        self.evaluate_model(model_name, best_model)
        print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")

    def train_lasso(self):
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'max_iter': [10000, 20000]
        }
        self.train_with_grid_search("Lasso", Lasso(), param_grid)

    def train_ridge(self):
        param_grid = {
            'alpha': [0.1, 1, 10, 100, 1000],
            'max_iter': [10000, 20000]
        }
        self.train_with_grid_search("Ridge", Ridge(), param_grid)

    def train_elastic_net(self):
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [10000, 20000]
        }
        self.train_with_grid_search("ElasticNet", ElasticNet(), param_grid)

    def train_svr(self):
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.01, 0.1, 1, 10]
        }
        self.train_with_grid_search("SVR", SVR(), param_grid)

    def train_decision_tree(self):
        param_grid = {
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 20]
        }
        self.train_with_grid_search("DecisionTree", DecisionTreeRegressor(random_state=self.random_state), param_grid)

    def train_random_forest(self):
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 20]
        }
        self.train_with_grid_search("RandomForest", RandomForestRegressor(random_state=self.random_state), param_grid)

    def train_gradient_boosting(self):
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9]
        }
        self.train_with_grid_search("GradientBoosting", GradientBoostingRegressor(random_state=self.random_state), param_grid)

    def train_knn(self):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }
        self.train_with_grid_search("KNN", KNeighborsRegressor(), param_grid)

    def evaluate_model(self, model_name, model):
        # Training error
        y_train_pred = model.predict(self.X_train)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        # Testing error
        y_test_pred = model.predict(self.X_test)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        self.metrics[model_name] = {
            "Train MSE": train_mse,
            "Train MAE": train_mae,
            "Train R2": train_r2,
            "Test MSE": test_mse,
            "Test MAE": test_mae,
            "Test R2": test_r2
        }
        self.model_scores[model_name] = test_r2
        print(f"Model {model_name} R²: {test_r2}")

    def find_best_model(self):
        best_model_name = max(self.model_scores, key=self.model_scores.get)
        best_metrics = self.metrics[best_model_name]
        return best_model_name, best_metrics
    
    def save_best_model(self):
        best_model_name = max(self.metrics, key=lambda k: self.metrics[k]["Test R2"])
        best_model = self.models[best_model_name]
        
        # Get the number of existing files in the directory
        directory = "data/models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        existing_files = os.listdir(directory)
        file_number = len(existing_files) + 1
        
        file_path = f"{directory}/{best_model_name}_{file_number}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Best model {best_model_name} saved to {file_path}")

    def visualize_metrics(self):
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        
        # Melt the DataFrame to have a long-form DataFrame suitable for seaborn
        metrics_melted_r2 = metrics_df.melt(id_vars="Model", value_vars=["Train R2", "Test R2"], 
                                            var_name="Metric", value_name="R2 Score")
        metrics_melted_mse = metrics_df.melt(id_vars="Model", value_vars=["Train MSE", "Test MSE"], 
                                             var_name="Metric", value_name="MSE")
        metrics_melted_mae = metrics_df.melt(id_vars="Model", value_vars=["Train MAE", "Test MAE"], 
                                             var_name="Metric", value_name="MAE")
        
        # Plot R2 Scores
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Model", y="R2 Score", hue="Metric", data=metrics_melted_r2)
        plt.title('Training and Testing R2 Scores for Trained Models')
        plt.xticks(rotation=45)
        plt.show()
        
        # Plot MSE
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Model", y="MSE", hue="Metric", data=metrics_melted_mse)
        plt.title('Training and Testing Mean Squared Error for Trained Models')
        plt.xticks(rotation=45)
        plt.show()
        
        # Plot MAE
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Model", y="MAE", hue="Metric", data=metrics_melted_mae)
        plt.title('Training and Testing Mean Absolute Error for Trained Models')
        plt.xticks(rotation=45)
        plt.show()

    def plot_predictions(self):
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        top_3_models = metrics_df.sort_values(by='Test R2', ascending=False).head(3)['Model']
        
        plt.figure(figsize=(12, 8))
        
        for model_name in top_3_models:
            model = self.models[model_name]
            y_pred = model.predict(self.X_test)
            plt.scatter(self.y_test, y_pred, alpha=0.5, label=model_name)
        
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values for Top 3 Models')
        plt.legend()
        plt.show()

    def plot_feature_importances(self):
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        top_3_models = metrics_df.sort_values(by='Test R2', ascending=False).head(3)['Model']
        
        for model_name in top_3_models:
            model = self.models[model_name]
            if hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            elif hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                print(f"Model {model_name} does not have feature importances or coefficients.")
                continue
            
            # Convert self.X to DataFrame to access column names
            feature_names = pd.DataFrame(self.X).columns
            feature_importances = pd.Series(importances, index=feature_names)
            feature_importances = feature_importances.sort_values(ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_importances.values, y=feature_importances.index)
            plt.title(f'Top 10 Feature Importances for {model_name}')
            plt.show()

    def run(self):
        self.split_data()
        # #self.train_linear_regression()
        self.train_ridge()
        self.train_lasso()
        self.train_elastic_net()
        self.train_svr()
        self.train_decision_tree()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_knn()
        best_model_name, best_metrics = self.find_best_model()
        print(f"Best model: {best_model_name}")
        print(f"Metrics: {best_metrics}")
        self.save_best_model()
        self.visualize_metrics()
        self.plot_predictions()
        self.plot_feature_importances()