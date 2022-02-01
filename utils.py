import numpy as np
import pickle
from PIL import Image
import tensorflow as tf


def get_data(path: str) -> dict:
    """
    Loads a dictionary of data.

    Parameters
    ----------
    path : string
        Path to the file

    Returns
    -------
    data : dict
        Dictionary containing:
            X_train, y_train, X_test, y_test, X_valid, y_valid
    """
    data = pickle.load(open(path, 'rb'))
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
    image = image.resize((100, 100))
    image = tf.cast(image, dtype=tf.float32)
    return image


def predict(model, photo):
    photoRes = preprocess_image(photo)
    predictions = model.predict(np.array([photoRes]))
    result = np.argmax(predictions)
    return result


def load_model(path: str) -> tf.keras.Model:
    """
    Loads a saved model.

    Parameters
    ----------
    path : string
        Path to the file

    Returns
    -------
    model : tf.keras.Model

    """
    model = tf.keras.models.load_model(path)
    return model
