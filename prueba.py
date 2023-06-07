import streamlit as st
import joblib
# Load the trained model
model = joblib.load('/Users/miguelpalospou/Desktop/IRONHACK/Projects/Final-project/trained_model/model.pkl')

# Define the Streamlit layout
st.title("Predictive Model")
st.write("Enter the required information and click 'Predict' to see the results.")

# Create input fields for user input
input1 = st.number_input("Input 1")
input2 = st.number_input("Input 2")
input3 = st.number_input("Input 3")

