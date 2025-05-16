from preprocessing import build_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
import joblib


def train_model(data, target, model=None, param_grid=None, num_imputer="mean",
                cat_imputer="most_frequent", cv=5, filename=None):
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
        
    predictions = best_model.predict(X_test)
        
    return {
        "best_model": best_model,
        "predictions": predictions,
        "best_params": best_params,
        "scores": scores,
        }
