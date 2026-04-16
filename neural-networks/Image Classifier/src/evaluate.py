from sklearn.metrics import accuracy_score

def evaluate_models(sk_model, dl_model, X_test, y_test):
    sk_pred = sk_model.predict(X_test.reshape(len(X_test), -1))
    print("Sklearn:", accuracy_score(y_test, sk_pred))
