import numpy as np
from keras.preprocessing import image
import os
import tensorflow as tf

class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename

    def predict(self):
        # load model
        model = tf.keras.models.load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(250, 250))
        test_image = test_image.convert('L')
        test_image = image.img_to_array(test_image)
        print("==================================", test_image.shape)
        # test_image = np.expand_dims(test_image, axis=0)
        probabilities = model.predict(test_image)[0]

        # Define class labels
        class_labels = ['Covid', 'Normal', 'Viral-Pneumonia']  

        predicted_class_index = np.argmax(probabilities)
        prediction = class_labels[predicted_class_index]

        # Get the predicted probability for the predicted class
        predicted_probability = probabilities[predicted_class_index]

        # Return the prediction and probabilities as a dictionary
        return [{"image": prediction, "probability": float(predicted_probability)}]
    
