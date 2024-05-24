import streamlit as st
import pickle
import numpy as np
from app_functions import *


gender_list = ['M','F']
areas_list = ['Southwest', 'Central', 'N Hollywood', 'Mission', 'Devonshire',
       'Northeast', 'Harbor', 'Van Nuys', 'West Valley', 'West LA',
       'Wilshire', 'Pacific', 'Rampart', '77th Street', 'Hollenbeck',
       'Southeast', 'Hollywood', 'Newton', 'Foothill', 'Olympic',
       'Topanga']
descent_list = ['B', 'H', 'W', 'A']
premis_list = ['Residential building', 'Street', 'Other inside', 'Store',
       'Restaurant/Bar', 'Parking', 'Public transport facilities',
       'Medical facility', 'Vehicle', 'Other outside', 'Park', 'School']

st.title("Identification of crime exposure using machine learning models.")


geo_coordinates_dict = {
   'Southwest': [34.0376, 33.9950, -118.2744, -118.3805],
   'Central': [34.0761, 34.0296, -118.2259, -118.274], 
   'N Hollywood': [34.2199, 34.117, -118.318, -118.4263], 
   'Mission': [34.3343, 34.0144, -118.2653, -118.5047], 
   'Devonshire': [34.333, 34.22, -118.4555, -118.6324], 
   'Northeast': [34.1586, 34.0687, -118.1677, -118.3282], 
   'Harbor': [34.0075, 33.7061, -118.2216, -118.3289], 
   'Van Nuys': [34.2149, 34.1264, -118.4053, -118.4738], 
   'West Valley': [34.2214, 33.9465, -118.2781, -118.5665], 
   'West LA': [34.1324, 34.0266, -118.3761, -118.5713], 
   'Wilshire': [34.089, 34.0319, -118.3174, -118.3897], 
   'Pacific': [34.0317, 33.9165, -118.3602, -118.482], 
   'Rampart': [34.0925, 34.0377, -118.2478, -118.3091], 
   '77th Street': [34.0038, 33.9382, -118.2564, -118.3586], 
   'Hollenbeck': [34.1115, 34.0128, -118.1554, -118.23], 
   'Southeast': [33.9602, 33.8729, -118.2279, -118.2918], 
   'Hollywood': [34.1346, 34.0575, -118.3004, -118.3955], 
   'Newton': [34.0407, 33.9747, -118.2235, -118.2809], 
   'Foothill': [34.294, 34.2065, -118.2658, -118.4415], 
   'Olympic': [34.0836, 34.0371, -118.2831, -118.3239], 
   'Topanga': [34.2427, 34.1356, -118.5622, -118.6676]
   }

form = st.empty()

if 'page' not in st.session_state:
    st.session_state.page = 'Form'

if st.session_state.page == 'Form':
   with form.container():
      gender = st.selectbox(
         "Choose gender",
         gender_list,
         index=0,
         placeholder="Indeterminate",
      )

      descent = st.selectbox(
         "Choose ethnicity descent: W - White, H - Hispanic and Latino, B - Black, A - Asian",
         descent_list,
         index=0,
         placeholder="Indeterminate",
      )

      age = st.number_input("Age", min_value=0, max_value=120, value=20)

      time = st.time_input(
         "Choose time", value="now", step=60)

      date = st.date_input("Choose date", value="default_value_today", format="YYYY/MM/DD")

      area = st.selectbox(
         "Choose area name (based on LAPD area stations)",
         areas_list,
         index=0,
         placeholder="Indeterminate",
      )

      geo_LAT = st.number_input(
         f"Type latitiude between {geo_coordinates_dict[area][0]} and {geo_coordinates_dict[area][1]}",
         max_value=geo_coordinates_dict[area][0],
         min_value=geo_coordinates_dict[area][1],
         step=0.001,
         format="%.4f")

      geo_LON = st.number_input(
         f"Type latitiude between {geo_coordinates_dict[area][2]} and {geo_coordinates_dict[area][3]}",
         max_value=geo_coordinates_dict[area][2],
         min_value=geo_coordinates_dict[area][3],
         step=0.001,
         format="%.4f")

      premis = st.selectbox(
         "Choose short description of area",
         premis_list,
         index=0,
         placeholder="Indeterminate",
      )

      if st.button('Predict'):
         pred = get_prediction(area, gender, descent, premis, time, age, geo_LAT, geo_LON, date)
         st.session_state.page = 'Prediction'


if st.session_state.page == 'Prediction':
   form.empty()
   st.markdown(f"<div align='center', style='font-size:20px'>{pred}</div>", unsafe_allow_html=True)