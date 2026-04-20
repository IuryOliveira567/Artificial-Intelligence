import tensorflow as tf
from tensorflow.keras import layers, Model


user_input = layers.Input(shape=(1,))
item_input = layers.Input(shape=(1,))

user_emb = layers.Embedding(n_users, 50)(user_input)
item_emb = layers.Embedding(n_items, 50)(item_input)

user_vec = layers.Flatten()(user_emb)
item_vec = layers.Flatten()(item_emb)

x = layers.Concatenate()([user_vec, item_vec])
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)

output = layers.Dense(1)(x)

model = Model([user_input, item_input], output)
model.compile(
    loss="mse",
    optimizer="adam",
    metrics=["mae"]
)

model.summary()
