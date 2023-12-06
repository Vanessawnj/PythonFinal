import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

st.title("Prediction of LinkedIn Usage")

# 定义收入范围的选项
income_options = {
    "Less than $10,000":1,
    "10 to under $20,000":2,
    "20 to under $30,000":3,
    "30 to under $40,000":4,
    "40 to under $50,000":5,
    "50 to under $75,000":6,
    "75 to under $100,000":7,
    "100 to under $150,000":8,
    "$150,000 or more":9
}
education_options = {
    "Less than high school (Grades 1-8 or no formal schooling)":1,
    "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":2,
    "High school graduate (Grade 12 with diploma or GED certificate)":3,
    "Some college, no degree (includes some community college)":4,
    "Two-year associate degree from a college or university":5,
    "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":6,
    "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":7,
    "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)":8
}

parent = st.selectbox("Are you a parent of a child under 18 living in your home?", options=["Yes", "No"])
income = st.selectbox("Select your household income:", list(income_options.keys()))
education = st.selectbox("Select your highest level of education:", list(education_options.keys()))
married = st.selectbox("Are you married?", options=["Yes", "No"])
female = st.selectbox("Are you a female?", options=["Yes", "No"])
age = st.number_input("Please enter your age:")

married = 1 if married == "Yes" else 0
parent = 1 if parent == "Yes" else 0
female = 1 if female == "Yes" else 0
income = income_options[income]
education = education_options[education]

# transform the inputs
model_inputs = np.array([[married,parent,income,education,female,age]])



if st.button("Predict!"):
    
    prediction = model.predict(model_inputs)
    p = model.predict_proba(model_inputs)[0][1]
    st.write("The prediction of whether the person is a LinkedIn user: ", "Yes" if prediction[0] == 1 else "No")
    st.write("The probability of using LinkedIn is:", p)