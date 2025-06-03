from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


def build_pipeline(data, model=None, num_imputer="mean", cat_imputer="most_frequent"):
    """
    Build a preprocessing and modeling pipeline.
    
    Parameters:
    - data: DataFrame with training data (including all features)
    - model: A scikit-learn estimator (default: LogisticRegression)
    - num_imputer: Numeric imputer method (default: mean)
    - cat_imputer: Categoric imputer method (default: most_frequent)

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
            ("imputer", SimpleImputer(strategy=num_imputer)),
            ("scaler", StandardScaler())
        ])
        
        preprocessor.transformers.append(("num", num_pipeline, num_cols))

    if(cat_imputer):
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=cat_imputer)),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ])

        preprocessor.transformers.append(("cat", cat_pipeline, cat_cols))

    if(preprocessor.transformers):
        pipeline.steps.insert(0, ("preprocessor", preprocessor))

    return pipeline
