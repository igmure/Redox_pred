import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

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
        self.scaler = StandardScaler()

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and test sets and normalize features.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        self.normalize_features()

    def normalize_features(self):
        """
        Normalize the features using StandardScaler.
        """
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

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
        
        # Print the best hyperparameters
        print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")

    def train_svr(self):
        """
        Train a Support Vector Regression model using grid search.
        """
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 1]
        }
        self.train_with_grid_search("SVR", SVR(), param_grid)

    def train_decision_tree(self):
        """
        Train a Decision Tree Regression model using grid search.
        """
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        self.train_with_grid_search("DecisionTree", DecisionTreeRegressor(), param_grid)

    def train_random_forest(self):
        """
        Train a Random Forest Regression model using grid search.
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        self.train_with_grid_search("RandomForest", RandomForestRegressor(), param_grid)

    def train_gradient_boosting(self):
        """
        Train a Gradient Boosting Regression model using grid search.
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        self.train_with_grid_search("GradientBoosting", GradientBoostingRegressor(), param_grid)

    def train_knn(self):
        """
        Train a K-Nearest Neighbors Regression model using grid search.
        """
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
        self.train_with_grid_search("KNN", KNeighborsRegressor(), param_grid)

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

    def display_trained_models(self):
        """
        Display the names of the trained models.
        """
        print("Trained models:")
        for model_name in self.models.keys():
            print(model_name)

    def visualize_metrics(self):
        """
        Visualize the metrics of the trained models.
        """
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        
        # Plot MSE
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='MSE', data=metrics_df)
        plt.title('Mean Squared Error of Trained Models')
        plt.xticks(rotation=45)
        plt.show()
        
        # Plot MAE
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='MAE', data=metrics_df)
        plt.title('Mean Absolute Error of Trained Models')
        plt.xticks(rotation=45)
        plt.show()
        
        # Plot R2 Score
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='R2', data=metrics_df)
        plt.title('R2 Score of Trained Models')
        plt.xticks(rotation=45)
        plt.show()

    def plot_predictions(self):
        """
        Plot predicted y values versus actual y values for each model.
        """
        plt.figure(figsize=(12, 8))
        
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            plt.scatter(self.y_test, y_pred, alpha=0.5, label=model_name)
        
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        plt.show()

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
            param_grid={"alpha": [0.01, 0.1, 1, 10, 100, 1000]}
        )
        
        # Perform grid search for Lasso Regression
        self.train_with_grid_search(
            "Lasso", 
            Lasso(), 
            param_grid={"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
        )
        
        # Train SVR
        self.train_svr()
        
        # Train Decision Tree Regression
        self.train_decision_tree()
        
        # Train Random Forest Regression
        self.train_random_forest()
        
        # Train Gradient Boosting Regression
        self.train_gradient_boosting()
        
        # Train K-Nearest Neighbors Regression
        self.train_knn()
        
        # Display trained models
        self.display_trained_models()
        
        # Find the best model
        best_model_name, best_metrics = self.find_best_model()
        print(f"Best model: {best_model_name}")
        print(f"Metrics: {best_metrics}")
        
        # Visualize metrics
        self.visualize_metrics()
        
        # Plot predictions
        self.plot_predictions()
