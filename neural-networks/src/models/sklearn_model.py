from sklearn.neural_network import MLPClassifier

def build_model():
    return MLPClassifier(hidden_layer_sizes=(100,))
