from preprocessing import build_pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import numpy as np
import joblib
import matplotlib.pyplot as plt


class Data_Training(object):
    """
    A class for training and evaluating machine learning models using pipelines
    with preprocessing for numerical and categorical features.
    """
    
    def __init__(self, data, model, train_test_data=None, target=None, ev_type="regression",
                 num_imputer="mean", cat_imputer="most_frequent"):
        """
        Initialize the Data_Training instance.

        Args:
            - data: Dataset object with a method split_train_test() returning train/test splits.
            - model: Scikit-learn compatible estimator (e.g., Ridge, SVR, etc.).
            - train_test_data: tuple containing the training and test sets(e.g, (x_train, x_test, y_train, y_test)
            - target: str, the target feature to be predicted
            - ev_type: problem type (regression, classification)
            - num_imputer: Strategy for imputing numerical data ('mean', 'median', etc.).
            - cat_imputer: Strategy for imputing categorical data ('most_frequent', etc.).
        """

        self.data = data
        self.model = model
        self.target = target
        self.ev_type = ev_type
        
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer
        
        if(target):
            train_set, test_set = self.data.split_train_test()

            self.X_train = train_set.drop(self.target, axis=1)
            self.Y_train = train_set[self.target]

            self.X_test = test_set.drop(self.target, axis=1)
            self.Y_test = test_set[self.target]
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_data
            
    def train_model(self, param_grid, cv=5, plot=False):

        """
        Train a model using a specified target feature.
    
        Parameters:
            - param_grid (dict): A dictionary with parameters names (`str`) as keys and lists of parameter settings to try.
            - cv: Int, number of cross-validation folds
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

            self.best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
              
            print("best parameters : ", best_params)
        else:
            self.best_model = pipeline.fit(self.X_train, self.Y_train)
    
        Y_pred = self.best_model.predict(self.X_test)
        self.evaluate(y_test=self.Y_test, prediction=Y_pred, model=self.best_model, plot=plot)

        return {
           "best_model": self.best_model,
           "predictions": Y_pred,
           "best_params": best_params,
           "scores": scores
        }

    def evaluate(self, y_test, prediction, model=None, plot=False):
        """
        Evaluate model predictions using common regression metrics.

        Args:
            - y_test (array-like): Ground truth target values.
            - prediction (array-like): Predicted target values.
            - evaluation_type: regression, classification
            - plot (bool, optional): Whether to plot results and residuals. Defaults to False.

        Prints:
            RMSE, MAE, and R² metrics.
        """

        if(self.ev_type == "regression"):
            rmse = np.sqrt(mean_squared_error(y_test, prediction))
            mae = mean_absolute_error(y_test, prediction)
            r2 = r2_score(y_test, prediction)

            print("Test RMSE:", rmse)
            print("Test MAE:", mae)
            print("Test R²:", r2)

            if(plot):
                residuals = y_test - prediction
                self.plot_result(y_test, prediction, residuals)
        elif(self.ev_type == "classification"):
            acc = accuracy_score(y_test, prediction)
            prec = precision_score(y_test, prediction, average="weighted", zero_division=0)
            
            rec = recall_score(y_test, prediction, average="weighted", zero_division=0)
            f1 = f1_score(y_test, prediction, average="weighted", zero_division=0)

            cm = confusion_matrix(y_test, prediction)

            print("Accuracy:", acc)
            print("Precision:", prec)
            print("Recall:", rec)
            print("F1 Score:", f1)
            print("Confusion Matrix:\n", cm)

            if(plot):
                y_scores = cross_val_predict(model, self.X_train, self.Y_train, cv=3,
                                             method="decision_function")
                 
                precisions, recalls, thresholds = precision_recall_curve(self.Y_train, y_scores)
                self.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        
    def plot_precision_recall_vs_threshold(self, precisions, recalls, thresholds):

        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Precision-Recall vs Threshold")
        plt.legend(loc="best")
    
        plt.show()
        
    def final_model(self, best_model, threshold=None, **model_params):
        """
        Train and cross-validate a final model with the provided parameters.

        Args:
           best_model: A Scikit-learn compatible model class (not an instance).
           threshold: Threshold to apply on decision_function for classification.
           **model_params: Keyword arguments for the model initialization.

        Returns:
           - For classification with threshold: (pipeline, predict_with_threshold)
           - For other cases: nothing
        """

        pipeline = build_pipeline(
            data=self.X_train,
            model=best_model(**model_params),
            num_imputer=self.num_imputer,
            cat_imputer=self.cat_imputer
        )

        pipeline.fit(self.X_train, self.Y_train)

        if self.ev_type == "regression":
            scores = cross_val_score(pipeline, self.X_train, self.Y_train, cv=5, scoring='r2')
            print("Average R² (CV):", scores.mean())
            
            y_pred_new = pipeline.predict(self.X_test)
            self.evaluate(self.Y_test, y_pred_new, model=best_model(**model_params), plot=False)
        elif self.ev_type == "classification":
            if threshold is not None:
                def predict_with_threshold(X, threshold):
                    scores = pipeline.decision_function(X)
                    return (scores > threshold).astype(int)
            
                y_pred_new = predict_with_threshold(self.X_test, threshold)
                self.evaluate(self.Y_test, y_pred_new, model=best_model(**model_params), plot=False)
                
                return pipeline, predict_with_threshold
            else:
                y_pred_new = pipeline.predict(self.X_test)
                self.evaluate(self.Y_test, y_pred_new, model=best_model(**model_params), plot=False)
        else:
            print("Unknown ev_type! Must be 'regression' or 'classification'.")

    def plot_roc_curve(self, scores, label=None):

        fpr, tpr, thresholds = roc_curve(self.Y_test, scores)    
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.show()

    def plot_confusion_matrix(self, model):
        """
        Plots the confusion matrix for the given model using training data.

        Args:
           model : estimator object
        """
         
        y_train_pred = cross_val_predict(model, self.X_train, self.Y_train, cv=3)
        conf_mx = confusion_matrix(self.Y_train, y_train_pred)

        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.show()
    
    def save_model(self, filename, pipeline):
        """
        Save a model given a filename and pipeline

        Args:
           - filename: Get the file name to save the best model, format: pkl
        """
        
        if(filename.split(".")[-1] != "pkl"):
            print("[-] File format must be pkl.")
            return
            
        joblib.dump(pipeline, filename)
        print("[+] Model saved!")
        
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
