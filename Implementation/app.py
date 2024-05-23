import streamlit as st
import pickle


def main():
    bg = """<div style='background-color:black; padding:13px'>
              <h1 style='color:white'>Loan Approval Prediction GUI</h1>
              <h2 style='color:grey'>Mohammed Aman</h2>
              <h3 style='color:grey'>2020A7PS0050U</h3>
       </div>"""
    st.markdown(bg, unsafe_allow_html=True)

    left, right = st.columns((2,2))
    Gender = left.selectbox('Gender', ('Male', 'Female'))
    Married = right.selectbox('Married', ('Yes', 'No'))
    Dependents = left.selectbox('Dependents', ('None', 'One', 'Two', 'Three'))
    Education = right.selectbox('Education', ('Graduate', 'Not Graduate'))
    Self_Employed = left.selectbox('Self-Employed', ('Yes', 'No'))
    ApplicantIncome = right.number_input('Applicant Income')
    CoapplicantIncome = left.number_input('Coapplicant Income')
    LoanAmount = right.number_input('Loan Amount')
    Loan_Amount_Term = left.number_input('Loan Tenor')
    Credit_History = right.number_input('Credit History', 0.0, 1.0)
    Property_Area = st.selectbox('Property Area', ('Semiurban', 'Urban', 'Rural'))
    Classifier_sel = st.selectbox('Classifier Type',('RFC','SVC'))
    button = st.button('Predict')


    # if button is clicked
    if button:
        # make prediction
        result = predict(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
                        CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Classifier_sel)
        st.success(f'You are {result} for the loan')

# # load the train model
# with open('train_model.pkl', 'rb') as pkl:
#     train_model = pickle.load(pkl)

def predict(gender, married, dependent, education, self_employed, applicant_income,
           coApplicantIncome, loanAmount, loan_amount_term, creditHistory, propertyArea, classifiersel): 
# load the train model
    # with open('train_model.pkl', 'rb') as pkl:
    #     train_model = pickle.load(pkl)
    # processing user input
    clfr = 1 if classifiersel == 'RFC' else 2
    gen = 0 if gender == 'Male' else 1
    mar = 0 if married == 'Yes' else 1
    dep = float(0 if dependent == 'None' else 1 if dependent == 'One' else 2 if dependent == 'Two' else 3)
    edu = 0 if education == 'Graduate' else 1
    sem = 0 if self_employed == 'Yes' else 1
    pro = 0 if propertyArea == 'Semiurban' else 1 if propertyArea == 'Urban' else 2
    Lam = loanAmount / 1000
    cap = coApplicantIncome / 1000
    # making predictions
    if clfr == 1:
        with open('train_modelrfc.pkl', 'rb') as pkl:
            train_modelrfc = pickle.load(pkl)
        prediction = train_modelrfc.predict([[gen, mar, dep, edu, sem, applicant_income, cap,
                                      Lam, loan_amount_term, creditHistory, pro]])
    elif clfr == 2:
        with open('train_modeldtc.pkl', 'rb') as pkl:
            train_modeldtc = pickle.load(pkl)
        prediction = train_modeldtc.predict([[gen, mar, dep, edu, sem, applicant_income, cap,
                                      Lam, loan_amount_term, creditHistory, pro]])
    verdict = 'Not Eligible' if prediction == 0 else 'Eligible'
    return verdict

if __name__ == '__main__':
    main()
