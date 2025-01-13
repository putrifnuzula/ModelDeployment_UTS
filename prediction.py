import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('XGB_model.pkl', "rb"))

def main():
    st.title('Churn Prediction (Mid Exam Model Deployment)')

    #user input
    CreditScore = st.number_input('Credit Score', min_value=350.0, max_value=850.0, step=0.1)
    Geography = st.radio('Country', ['France', 'Germany', 'Spain'])
    Gender = st.selectbox('Gender', ['Female', 'Male'])
    Age = st.number_input('Age', min_value=18, max_value=92, step=1)
    Tenure = st.number_input("Duration of your bank account (in years)", min_value=0, max_value=10, step=1)
    Balance = st.number_input('Account Balance', min_value=0.0, max_value=250898.09, step=0.01)
    NumOfProducts = st.number_input('Number of Purchased Products', min_value=1, max_value=4, step=1)
    HasCrCard = st.selectbox('Have a Credit Card', ['No', 'Yes'])
    IsActiveMember = st.radio('Is an Active Member', ['No', 'Yes'])
    EstimatedSalary = st.number_input('Salary', min_value=11.58, max_value=199992.48, step=0.01)

    if st.button('Make Prediction'):
        #encode input
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
    #predict
    prediction = model.predict(features)
    return prediction[0]

if __name__ == '__main__':
    main()
