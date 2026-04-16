from tensorflow.keras.datasets import mnist


def load_data():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, X_test, y_train, y_test
