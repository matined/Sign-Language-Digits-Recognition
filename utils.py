import numpy as np
import pickle
from PIL import Image
import tensorflow as tf


def get_data() -> dict:
    """
    Loads a dictionary of data.

    Returns
    -------
    data : dict
        Dictionary containing:
            X_train, y_train, X_test, y_test, X_valid, y_valid
    """
    data = pickle.load(open('data/data.pkl', 'rb'))
    return data


def preprocess_image(filepath: str) -> tf.Tensor:
    """
    Process an image so it's ready for prediction.

    Parameters
    ----------
    filepath : string
        Path to the photo

    Returns
    -------
    image : tf.Tensor
        The image as a tensorflow tensor
    """
    image = Image.open(filepath)
    image.resize((100, 100))
    image = tf.cast(image, dtype=tf.float32)
    return image


def predict(model, photo):
    photoRes = preprocess_image(photo)
    predictions = model.predict(np.array([photoRes]))
    result = np.argmax(predictions)
    return result


def load_model():
    model = tf.keras.models.load_model('models/model_1.h5')
    return model
