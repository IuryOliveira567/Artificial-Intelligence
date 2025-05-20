from preprocessing import build_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import matplotlib.pyplot as plt


class Data_Training(object):
    """
    A class for training and evaluating machine learning models using pipelines
    with preprocessing for numerical and categorical features.
    """
    
    def __init__(self, data, target, model, num_imputer="mean", cat_imputer="most_frequent"):
        """
        Initialize the Data_Training instance.

        Args:
            - data: Dataset object with a method split_train_test() returning train/test splits.
            - target: str, the target feature to be predicted
            - model: Scikit-learn compatible estimator (e.g., Ridge, SVR, etc.).
            - num_imputer: Strategy for imputing numerical data ('mean', 'median', etc.).
            - cat_imputer: Strategy for imputing categorical data ('most_frequent', etc.).
        """

        self.data = data
        self.model = model
        self.target = target
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer

        train_set, test_set = self.data.split_train_test()
        
        self.X_train = train_set.drop(self.target, axis=1)
        self.Y_train = train_set[self.target]

        self.X_test = test_set.drop(self.target, axis=1)
        self.Y_test = test_set[self.target]
        
    def train_model(self, param_grid, cv=5, filename=None, plot=False):

        """
        Train a model using a specified target feature.
    
        Parameters:
            - param_grid (dict): A dictionary with parameters names (`str`) as keys and lists of parameter settings to try.
            - cv: Int, number of cross-validation folds
            - filename: Get the file name to save the best model, format: pkl
            - plot: plot the result graph

        Returns:
           dict: A dictionary containing:
              - 'best_model': The best trained model (Pipeline).
              - 'predictions': Predictions on the test set.
              - 'best_params': Best hyperparameters found by GridSearchCV (if applicable).
              - 'scores': RMSE from GridSearchCV or cross-validation.
        """
        
        scores = None
        best_params = None
    
        pipeline = build_pipeline(
            data=self.X_train,
            model=self.model,
            num_imputer=self.num_imputer,
            cat_imputer=self.cat_imputer
        )

        if(param_grid):
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="neg_mean_squared_error", cv=cv)
            grid_search.fit(self.X_train, self.Y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            scores = np.sqrt(-grid_search.best_score_)

            print("best parameters : ", best_params)
            print("best score : ", scores)
        else:
            self.best_model = pipeline.fit(self.X_train, self.Y_train)
            cv_scores = cross_val_score(pipeline, self.X_train, self.Y_train, scoring="neg_mean_squared_error", cv=cv)

            scores = np.sqrt(-cv_scores)
            print("score : ", scores)
    
        if(filename):
           joblib.dump(best_model, filename)
        
        Y_pred = best_model.predict(self.X_test)
        self.evaluate(self.Y_test, Y_pred, plot)

        return {
           "best_model": best_model,
           "predictions": Y_pred,
           "best_params": best_params,
           "scores": scores,
        }

    def evaluate(self, y_test, prediction, plot=False):
        """
        Evaluate model predictions using common regression metrics.

        Args:
            y_test (array-like): Ground truth target values.
            prediction (array-like): Predicted target values.
            plot (bool, optional): Whether to plot results and residuals. Defaults to False.

        Prints:
            RMSE, MAE, and R² metrics.
        """
 
        test_rmse = mean_squared_error(y_test, prediction)
        test_mae = mean_absolute_error(y_test, prediction)
        
        test_r2 = r2_score(y_test, prediction)
        residuals = y_test - prediction

        print("Test RMSE:", test_rmse)
        print("Test MAE:", test_mae)
        print("Test R²:", test_r2)

        if(plot):
            self.plot_result(y_test, prediction, residuals)

    def final_model(self, best_model, **model_params):
       """
       Train and cross-validate a final model with the provided parameters.

       Args:
          best_model: A Scikit-learn compatible model class (not an instance).
          **model_params: Keyword arguments for the model initialization.

       Prints:
          Average R² score across CV folds.
       """
       
       pipeline = build_pipeline(
           data=self.X_train,
           model=best_model(**model_params),
           num_imputer=self.num_imputer,
           cat_imputer=self.cat_imputer
       )
                
       pipeline.fit(self.X_train, self.Y_train)
       scores = cross_val_score(pipeline, self.X_train, self.Y_train, cv=5, scoring='r2')

       print("Average R² (CV):", scores.mean())    
       
    def plot_result(self, Y_test, Y_pred, residuals):
        """
        Plot predicted vs actual values and the distribution of residuals.

        Args:
            Y_test (array-like): Actual target values.
            Y_pred (array-like): Predicted target values.
            residuals (array-like): Difference between actual and predicted values.
        """

        #Scatter: Y_test vs Y_pred
        plt.figure(figsize=(6, 6))
        plt.scatter(Y_test, Y_pred, alpha=0.5)
        plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
        plt.xlabel("Real Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Real")
        plt.grid(True)
        plt.show()

        #Residuals
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=50)
        plt.title("Error Distribution")
        plt.xlabel("Error (Y_test - Y_pred)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
