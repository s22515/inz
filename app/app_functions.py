import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,  MinMaxScaler
from xgboost import XGBClassifier
import datetime
import joblib


def load_label_encoder() -> LabelEncoder:
    with open('models/label_encoder.pkl','rb') as f:
        label_encoder = joblib.load(f)
    return label_encoder

def load_one_hot_encoder() -> OneHotEncoder:
    with open('models/one_hot_encoder.pkl','rb') as f:
        one_hot_encoder = joblib.load(f)
    return one_hot_encoder

def load_min_max_scaler() -> MinMaxScaler:
    with open('models/min_max_scaler.pkl','rb') as f:
        min_max_scaler = joblib.load(f)
    return min_max_scaler

def load_xgb_model() -> XGBClassifier:
    with open('models/xgb_model_compressedt.pkl','rb') as f:
        xgb_model = joblib.load(f)
    return xgb_model

def convert_time(time: datetime.time) -> int:
    #converting time from ex. 12:00 to int 1200
    time_converted = str(time.hour) + str(time.minute)
    return int(time_converted)

def get_prediction(
    area: str, 
    gender: str, 
    descent: str, 
    premis: str, 
    time: datetime.time, 
    age: int, 
    geo_LAT: float,
    geo_LON: float,
    date: datetime.date) -> str:
    
    label_encoder = load_label_encoder()
    one_hot_encoder = load_one_hot_encoder()
    min_max_scaler = load_min_max_scaler() 
    xgb_model = load_xgb_model()

    categorical_to_encode = [area, gender, descent, premis]
    categorical_encoded = one_hot_encoder.transform([categorical_to_encode])

    time = convert_time(time)

    numeric_to_min_max_scaling = [time, age, geo_LAT, geo_LON, date.day, date.month]
    numeric_scaled = min_max_scaler.transform([numeric_to_min_max_scaling])
    prediction_input = np.concatenate((numeric_scaled, categorical_encoded), axis=1)

    xgb_prediction = xgb_model.predict_proba(prediction_input)
    sorted_preds = np.sort(xgb_prediction[0])[::-1][:3]
    top_preds = np.argsort(xgb_prediction[0])[-3:][::-1]

    predicted_class0 = label_encoder.inverse_transform([top_preds[0]])
    predicted_class1 = label_encoder.inverse_transform([top_preds[1]])
    predicted_class2 = label_encoder.inverse_transform([top_preds[2]])

    rounded1 = "{:.2f}".format(sorted_preds[0])
    rounded2 = "{:.2f}".format(sorted_preds[1])
    rounded3 = "{:.2f}".format(sorted_preds[2])

    text =f"""If you become a victim of a crime, it will be: <ul style='list-style:none'>
	<li style='font-size:20px'><b><span style='color: red'>{predicted_class0[0]}</span></b> with a probability of <b><span style='color: red;'>{rounded1}</sapn></b>% </li>
	<li style='font-size:20px'><b><span style='color: red'>{predicted_class1[0]}</span></b> with a probability of <b><span style='color: red'>{rounded2}</span></b>% </li>
	<li style='font-size:20px'><b><span style='color: red'>{predicted_class2[0]}</span></b> with a probability of <b><span style='color: red'>{rounded3}</span></b>% </li>
    <ul>"""
    return text



