import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

model = joblib.load('2602148814.pkl')

def main():
    st.title('Machine Learning Model Deployment')

    credit_score = st.number_input('Credit Score', min_value=0, max_value=850)
    age = st.number_input('Age', min_value=18, max_value=100)
    balance = st.number_input('Balance', min_value=0.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

    geography = st.selectbox('Geography', ['Spain', 'France', 'Germany'])  
    gender = st.selectbox('Gender', ['Male', 'Female'])  
    tenure = st.selectbox('Tenure', ['0','1','2','3','4','5','6','7','8','9','10'])  
    num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, step=1)
    has_cr_card = st.selectbox('Has Credit Card?', ['Yes', 'No']) 
    is_active_member = st.selectbox('Is Active Member?', ['Yes', 'No'])  

    if st.button('Make Prediction'):
        label_encoder = LabelEncoder()
        geography_encoded = label_encoder.fit_transform([geography])[0]
        gender_encoded = label_encoder.fit_transform([gender])[0]
        tenure_encoded = label_encoder.fit_transform([tenure])[0]
        has_cr_card_encoded = label_encoder.fit_transform([has_cr_card])[0]
        is_active_member_encoded = label_encoder.fit_transform([is_active_member])[0]

        features = [
            credit_score, geography_encoded, gender_encoded, age, tenure_encoded, balance, 
            num_of_products, has_cr_card_encoded, is_active_member_encoded, estimated_salary
        ]

        result = make_prediction(features)

        if result == 0:
            output_text = "Customer not likely to churn" 
        else:
            output_text = "Customer likely to churn"

        st.success(f'The prediction is: {output_text}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
