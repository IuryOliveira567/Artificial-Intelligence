from src.models.sklearn_model import build_model as build_sklearn
from src.models.keras_model import build_model as build_keras


def train_sklearn(X, y):
    model = build_sklearn()

    X = X.reshape(len(X), -1)
    model.fit(X.reshape(len(X), -1), y)

    return model

def train_keras(X, y):
    model = build_keras()
    
    model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(X, y, epochs=10, validation_split=0.1)
    
    return model
