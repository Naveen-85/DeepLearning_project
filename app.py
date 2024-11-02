import streamlit as st
import numpy as np
import tensorflow as tf 
from  sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd 
import pickle
model=tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('one_hot.pkl', 'rb') as file:
    one_hot=pickle.load(file)
with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)

st.title('cutomer churn ptrediction')

# User input fields
geography = st.selectbox("Geography", one_hot.categories_[0])  # Assuming 'onehot_encoder_geo' is predefined
gender = st.selectbox("Gender", label_encoder_gender.classes_)  # Assuming 'label_encoder_gender' is predefined
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],  # Assuming a label encoder is set up
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Use one_hot encoder to transform the value (it needs to be in a 2D array)
geo_encoded = one_hot.transform([['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display the result
if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")


input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display the result
if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")