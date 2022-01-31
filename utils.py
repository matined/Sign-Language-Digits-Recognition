import numpy
import pickle
import os


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
