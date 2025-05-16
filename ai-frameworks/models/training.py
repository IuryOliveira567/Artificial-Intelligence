from preprocessing import build_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import matplotlib.pyplot as plt


def train_model(data, target, model=None, param_grid=None, num_imputer="mean",
                cat_imputer="most_frequent", cv=5, filename=None, plot=False):
    """
    Train a model using a specified target feature.
    
    Parameters:
    - data: DataFrame containing the dataset (including target feature)
    - target: str, the target feature to be predicted
    - model: Scikit-learn estimator (optional, default: LinearRegression)
    - param_grid: dict, hyperparameters to test with GridSearchCV
    - num_imputer_strategy: Strategy for numeric imputers ('mean', 'median', 'most_frequent', 'constant')
    - cat_imputer_strategy: Strategy for categorical imputers ('most_frequent', 'constant')
    - cv: Int, number of cross-validation folds
    - filename: Get the file name to save the best model, format: pkl

    Returns:
    - dict: A dictionary containing:
        - 'predictions': Predictions on the test set.
        - 'scores': RMSE scores from cross-validation or GridSearchCV.
        - 'best_params': Best parameters found by GridSearchCV (if applicable).
    """

    scores = None
    best_params = None
    
    train_set, test_set = data.split_train_test()
        
    X_train = train_set.drop(target, axis=1)
    Y_train = train_set[target]

    X_test = test_set.drop(target, axis=1)
    Y_test = test_set[target]
    
    pipeline = build_pipeline(
        data=X_train,
        model=model,
        num_imputer=num_imputer,
        cat_imputer=cat_imputer
    )

    if(param_grid):
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="neg_mean_squared_error", cv=cv)
        grid_search.fit(X_train, Y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        scores = np.sqrt(-grid_search.best_score_)
        print("best parameters : ", best_params)
        print("best score : ", scores)
    else:
        best_model = pipeline.fit(X_train, Y_train)
        
        cv_scores = cross_val_score(pipeline, X_train, Y_train, scoring="neg_mean_squared_error", cv=cv)
        scores = np.sqrt(-cv_scores)
        print("score : ", scores)
    
    if(filename):
        joblib.dump(best_model, filename)
        
    Y_pred = best_model.predict(X_test)
    evaluate(Y_test, Y_pred, plot)

    return {
        "best_model": best_model,
        "predictions": Y_pred,
        "best_params": best_params,
        "scores": scores,
        }

def plot_result(Y_test, Y_pred, residuals):
    
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
        
def evaluate(y_test, prediction, plot=False):

    test_rmse = mean_squared_error(y_test, prediction)
    test_mae = mean_absolute_error(y_test, prediction)
    test_r2 = r2_score(y_test, prediction)
    residuals = y_test - prediction

    print("Test RMSE:", test_rmse)
    print("Test MAE:", test_mae)
    print("Test R²:", test_r2)

    if(plot):
      plot_result(y_test, prediction, residuals)
        
    

    
