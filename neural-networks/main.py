from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_sklearn, train_keras
from src.evaluate import evaluate_models


X_train, X_test, y_train, y_test = load_data()
X_train, X_test = preprocess_data(X_train, X_test)

sk_model = train_sklearn(X_train, y_train)
dl_model = train_keras(X_train, y_train)

evaluate_models(sk_model, dl_model, X_test, y_test)
