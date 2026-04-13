import os
from tensorflow.keras.datasets import mnist

def load_data():
    path = "raw/mnist.npz"
    
    if os.path.exists(path):
        return mnist.load_data(path)
    else:
        data = mnist.load_data()
        return data
