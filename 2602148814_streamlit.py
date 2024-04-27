import streamlit as st
import joblib
import numpy as np

# Load the machine learning model
model = joblib.load('RF_class.pkl')

def main():
    st.title('Churn Prediction Model Deployment')

    # Add user input components for features
    credit_score = st.slider('Credit Score', min_value=300, max_value=850, value=300)
    geography = st.selectbox('Geography', [0, 1, 2], format_func=convert_geography_label)
    gender = st.selectbox('Gender', [0, 1], format_func=convert_gender_label)
    age = st.slider('Age', min_value=18, max_value=100, value=18)
    tenure = st.slider('Tenure', min_value=0, max_value=10, value=0)
    balance = st.slider('Balance', min_value=0.0, max_value=250000.0, value=0.0)
    num_of_products = st.slider('Number of Products', min_value=1, max_value=4, value=1)
    has_cr_card = st.radio('Has Credit Card', [0, 1])
    is_active_member = st.radio('Is Active Member', [0, 1])
    estimated_salary = st.slider('Estimated Salary', min_value=0.0, max_value=200000.0, value=0.0)
    
    if st.button('Make Prediction'):
        features = [credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]
        result = make_prediction(features)
        if result == 1:
            st.error('The customer is likely to churn.')
        else:
            st.success('The customer is not likely to churn.')

def make_prediction(features):
    # Use the loaded model to make predictions
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Function to convert gender labels
def convert_gender_label(label):
    if label == 0:
        return 'Male'
    elif label == 1:
        return 'Female'

# Function to convert geography labels
def convert_geography_label(label):
    if label == 0:
        return 'Germany'
    elif label == 1:
        return 'France'
    else:
        return 'Spain'

if __name__ == '__main__':
    main()