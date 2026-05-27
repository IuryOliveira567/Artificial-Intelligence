import tensorflow as tf
from tensorflow import keras
from preprocessing import load_and_preprocess
from sklearn.metrics import classification_report
from evaluate import plot_history


X_train, X_test, y_train, y_test, scaler = load_and_preprocess("../data/sensor_data.csv")

model = keras.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal accuracy : {accuracy:.4f}")

predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

print("\nReport:\n")
print(classification_report(y_test, predicted_classes))

plot_history(history)
