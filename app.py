import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, encode_value
from PIL import Image


model = joblib.load(r'rf_tuned_model.joblib')

st.set_page_config(page_title="Road Traffic Accident Severity Prediction",
                   page_icon="🛣️", layout="wide")

features = ['Cause_of_accident','Day_of_week','Type_of_vehicle','Day','Area_accident_occured','Number_of_vehicles_involved','Number_of_casualties','Driving_experience','Types_of_Junction','Light_conditions']

options_cause_acc = ['Changing lane', 'Driving at high speed', 'Driving carelessly', 'Driving to the left', 'Driving under the influence of drugs', 'Drunk driving', 'Getting off the vehicle improperly', 'Improper parking', 'Moving Backward', 'No distancing', 'No priority to pedestrian', 'No priority to vehicle', 'Other', 'Overloading', 'Overspeed', 'Overtaking', 'Overturning', 'Turnover']

options_day_of_week = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

options_day_time = ['Dawn','Morning','Afternoon','Evening']

options_vehicle_type = ['Automobile', 'Bajaj', 'Bicycle', 'Lorry', 'Motorcycle', 'Other', 'Pick up upto 10Q', 'Public', 'Ridden horse', 'Special vehicle', 'Stationwagen', 'Taxi', 'Turbo']

options_accident_area = ['Church areas', 'Hospital areas', 'Industrial areas', 'Market areas', 'Office areas', 'Office areas', 'Other', 'Outside rural areas', 'Recreational areas', 'Recreational areas', 'Residential areas', 'Rural village areas', 'School areas']

options_driving_exp = ['1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'Below 1yr', 'No Licence']

options_junction_types = ['Crossing', 'No junction', 'O Shape', 'Other', 'T Shape', 'X Shape', 'Y Shape']

options_lighting_conditions = ['Daylight', 'Darkness']

st.markdown("<h1 style='text-align: center;'>🛣️ Road Traffic Accident Severity Prediction 🛣️</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):
        
        st.subheader("Enter the situation of the accident:")
        
        accident_cause = st.selectbox("What is the cause of the accident? ", options=options_cause_acc)
        casualties = st.slider("How many are the number of casualties involved? ", 1, 8, value=0, format="%d")
        day_of_week = st.selectbox("What day did the accident occur?: ", options=options_day_of_week)
        day_time = st.selectbox("At what time of the day did the accident occur? ", options=options_day_time)
        accident_area = st.selectbox("Where did the accident occur? ", options=options_accident_area)
        juntion_type = st.selectbox("What is the junction type of the road? ", options=options_junction_types)
        lighting_cond = st.selectbox("What is the lighting condition of the area? ", options=options_lighting_conditions)
        vehicles_involved = st.slider("How many vehicles are involved? ", 1, 7, value=0, format="%d")
        vehicle_type = st.selectbox("What is the vehicle type? ", options=options_vehicle_type)
        driving_experience = st.selectbox("How many years of driving experience does the driver has? ", options=options_driving_exp)
        
        submit = st.form_submit_button("Predict")


    if submit:
        accident_cause = encode_value(accident_cause,options_cause_acc)
        day_of_week = encode_value(day_of_week,options_day_of_week)
        day_time = encode_value(day_time,options_day_time)
        accident_area =  encode_value(accident_area,options_accident_area)
        juntion_type =  encode_value(juntion_type,options_junction_types)
        lighting_cond = encode_value(lighting_cond,options_lighting_conditions) 
        vehicle_type = encode_value(vehicle_type,options_vehicle_type)
        driving_experience = encode_value(driving_experience,options_driving_exp)

        data = np.array([accident_cause,casualties,day_of_week,day_time,accident_area,
                         juntion_type,lighting_cond,vehicles_involved,vehicle_type,driving_experience]).reshape(1,-1)
        #st.write(data)
        pred = get_prediction(data=data, model=model)
        
        if pred[0]==0:
            severity='Fatal Injury'
        elif pred[0]==1:
            severity='Serious_Injury'
        else:
            severity='Slight Injury'

        st.write(f"⚠️ The accident severity prediction is:  {severity} ⚠️")

if __name__ == '__main__':
    main()