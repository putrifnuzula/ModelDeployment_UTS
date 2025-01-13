import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('XGB_model.pkl', "rb"))

def main():
    st.title('Churn Prediction (Mid Exam Model Deployment)')

    # Add user input components for 10 features
    CreditScore = st.slider('Credit Score', 350.0, 850.0)
    Geography = st.radio('Country', ['France', 'Germany', 'Spain'])
    Gender = st.selectbox('Gender', ['Female', 'Male'])
    Age = st.number_input('Age', 18, 92)
    Tenure = st.number_input("Duration of your bank account (in years)", 0, 10)
    Balance = st.slider('Account Balance', 0.0, 250898.09)
    NumOfProducts = st.number_input('Number of Purchased Products', 1, 4)
    HasCrCard = st.selectbox('Have a Credit Card', ['No', 'Yes'])
    IsActiveMember = st.radio('Is an Active Member', ['No', 'Yes'])
    EstimatedSalary = st.slider('Salary', 11.58, 199992.48)

    if st.button('Make Prediction'):
        #encode
        Geography_encoded = {'France': 0, 'Germany': 1, 'Spain': 2}[Geography]
        Gender_encoded = {'Female': 0, 'Male': 1}[Gender]
        HasCrCard_encoded = {'No': 0, 'Yes': 1}[HasCrCard]
        IsActiveMember_encoded = {'No': 0, 'Yes': 1}[IsActiveMember]
        features = [
            CreditScore, 
            Geography_encoded, 
            Gender_encoded, 
            Age, 
            Tenure, 
            Balance, 
            NumOfProducts, 
            HasCrCard_encoded, 
            IsActiveMember_encoded, 
            EstimatedSalary
        ]

        input_array = np.array(features).reshape(1, -1)
        result = make_prediction(input_array)
        st.success(f'The prediction is: {"Churn" if result == 1 else "Not Churn"}')

def make_prediction(features):
    #predict model
    prediction = model.predict(features)
    return prediction[0]

if __name__ == '__main__':
    main()
