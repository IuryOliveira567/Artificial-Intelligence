from PIL import Image
import numpy as np

def preprocess_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((28, 28))
    
    img = np.array(img)

    img = 255 - img

    img = img / 255.0

    return img

def predict_image(model, path):
    img = preprocess_image(path)
    
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    
    predicted_class = np.argmax(prediction)

    return predicted_class
