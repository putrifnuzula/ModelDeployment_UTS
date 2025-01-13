import streamlit as st
import joblib
import numpy as np
import xgboost
import pickle
# Read the machine learning model 
# model = joblib.load('XGB_model.pkl')
model = pickle.load(open('XGB_model.pkl', "rb"))



def main():
    st.title('Churn Prediction (Mid Exam Model Deployment)')

    # Add user input components for 10 features
    CreditScore = st.slider('Credit Score', 350.0, 850.0)
    Geography = st.radio('Country', ['France', 'Germany', 'Spain'])
    Gender = st.selectbox('Gender', [0,1])
    Age = st.number_input('Age', 18.0,92.0)
    Tenure = st.number_input("Duration of your bank account (in years)", 0,10)
    Balance = st.slider('Account Balance', 0.0, 250898.09)
    NumOfProducts = st.number_input('Number of Purchased Products', 1, 4)
    HasCrCard = st.selectbox('Have a Credit Card', [0,1])
    IsActiveMember = st.radio('Is an Active Member', ['0' ,'1'])
    EstimatedSalary = st.slider('Salary', 11.58, 199992.48)

    if st.button('Make Prediction'):
        features = [CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
        # ft = feat_list = np.array(features, dtype=object)
        result = make_prediction(ft)
        st.success(f'The prediction is: {result}')


def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
