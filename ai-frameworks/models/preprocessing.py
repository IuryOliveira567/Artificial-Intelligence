from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import time
from sklearn.linear_model import LogisticRegression


def build_pipeline(data, num_imputer, num_imputer_method, num_scaler,
                   cat_imputer, cat_imputer_method, model=None):
    """
    Build a preprocessing and modeling pipeline.
    
    Parameters:
    - data: DataFrame with training data (including all features)
    - num_imputer: Numeric imputer function (default: SimpleImputer)
    - num_imputer_method: Numeric imputer method (default: mean)
    - num_scaler: Numeric scaler function (default: SimpleScaler)
    - cat_imputer: Categorical imputer method (default: SimpleImputer)
    - cat_imputer_method: Categoric imputer method (default: most_frequent)
    - model: A scikit-learn estimator (default: LogisticRegression)

    Returns:
    - pipeline: a scikit-learn Pipeline object
    """

    if model is None:
        model = LogisticRegression()

    pipeline = Pipeline([
        ("model", model)
    ])
    
    preprocessor = ColumnTransformer([])

    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if(num_imputer):
        num_pipeline = Pipeline([
            ("imputer", num_imputer(strategy=num_imputer_method)),
            ("scaler", num_scaler())
        ])
        
        preprocessor.transformers.append(("num", num_pipeline, num_cols))
    
    if(cat_imputer):
        cat_pipeline = Pipeline([
            ("imputer", cat_imputer(strategy=cat_imputer_method)),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ])

        preprocessor.transformers.append(("cat", cat_pipeline, cat_cols))

    if(preprocessor.transformers):
        pipeline.steps.insert(0, ("preprocessor", preprocessor))
    
    return pipeline
