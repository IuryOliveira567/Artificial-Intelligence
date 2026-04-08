def train_sklearn(X, y):
    model = build_sklearn()
    model.fit(X.reshape(len(X), -1), y)
    return model
