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

    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=num_imputer)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=cat_imputer)),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline
