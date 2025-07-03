import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import streamlit as st



# Load the pre trained model
model = tf.keras.models.load_model('model.h5')


# Load the encoders and scalers

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geography.pkl', 'rb') as file:
    one_hot_encoder_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="PredictableAI")
st.title('Predict the unpredictable Customer Churn')

# Taking input from user
geography = st.selectbox('Geography', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,100)
balance = st.number_input('Balance', min_value=0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850)
estimated_salary = st.number_input('Estimated Salary', min_value=0)
tenure = st.slider('Tenure', 0, 10)
number_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [number_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary],
})


# One-hot encode the Geography column
geo_encoded = one_hot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns= one_hot_encoder_geography.get_feature_names_out(['Geography']))


# Concatenate the one-hot encoded Geography with the input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)


# Display the prediction
# add a predict button after it should display
predict_button = st.button('Predict Churn')
status = st.empty()
if predict_button:
    status.text('Processing your request...')
    # Predict the churm
    prediction = model.predict(input_data_scaled)
    # Display the result
    if prediction[0][0] > 0.5:
        status.empty()
        st.success('The customer is likely to churn with a probability of {:.2f}%'.format(prediction[0][0] * 100))
    else:
        status.empty()
        st.success('The customer is not likely to churn with a probability of {:.2f}%'.format((1 - prediction[0][0]) * 100))
