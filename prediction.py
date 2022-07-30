import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

le = joblib.load(r'label_encoder.joblib')

def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)

def encode_value(entered_value,options):
    values = list(le.fit_transform(options))
    keys = options
    dictionary = dict(zip(keys, values))
    encoded_value = dictionary[entered_value]
    return encoded_value
        