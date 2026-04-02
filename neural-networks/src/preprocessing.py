def preprocess_data(X_train, X_test):

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, X_test
