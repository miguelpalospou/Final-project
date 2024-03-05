import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle  

st.title('FLIP-FLAP')
st.markdown('''
A real-time house price analyzer. Are you a potential buyer? 
Check out the fair price for your target house. 
Are you a seller? Find out the price you should sell your flat.
''')



# Load the trained model
# Define the Streamlit layout
st.title("Predictive Model")
st.write("Enter the required information and click 'Predict' to see the results.")

# Create input fields for user input
rooms = st.slider('number of rooms', 1, 10, 3)
area = st.number_input("Square Meters", step=1, value=0)
plant = st.number_input("Floor", step=1, value=0, min_value=0, max_value=7, format="%d")
lift_options = ['yes', 'no']
lift_lift = st.selectbox("lift", lift_options)

parking_options = ['yes', 'no']
parking_yes = st.selectbox("parking", parking_options)


neighborhoods = [
"la Dreta de l'Eixample", 'la Vila Olímpica del Poblenou',
       'el Poble-sec', 'Diagonal Mar i el Front Marítim del Poblenou',
       'el Barri Gòtic', "l'Antiga Esquerra de l'Eixample",
       'Sant Pere, Santa Caterina i la Ribera', 'la Sagrada Família',
       'el Raval', 'el Poblenou', 'Sant Gervasi - Galvany',
       'el Putxet i el Farró', 'la Barceloneta', 'Sants', 'OTHER',
       'Sant Antoni', 'la Vila de Gràcia', 'les Tres Torres',
       'el Fort Pienc', "la Nova Esquerra de l'Eixample", 'Pedralbes',
       'la Maternitat i Sant Ramon', "el Camp d'en Grassot i Gràcia Nova",
       'el Baix Guinardó', 'Sarrià', 'les Corts', 'el Guinardó',
       'Sant Gervasi - la Bonanova', "el Camp de l'Arpa del Clot",
       'el Congrés i els Indians']
selected_neighborhood = st.selectbox("Neighborhood", neighborhoods)

predict_button = st.button("Predict")




if predict_button:
    # Create a DataFrame with user inputs
    with open('trained_model/model_2.pkl', 'rb') as f:
        model = pickle.load(f)

    data = pd.DataFrame({
        'rooms': [rooms],
        'area': [area],
        'plant': [plant],
        'lift_lift': [1 if lift_lift == 'yes' else 0],
        'parking_yes': [1 if parking_yes == 'yes' else 0]
    })

    for neighborhood in neighborhoods:
        if neighborhood in selected_neighborhood:
            data[f'neighbourhood_{neighborhood}'] = 1
        else:
            data[f'neighbourhood_{neighborhood}'] = 0

    original_columns_order = [
    'area', 'rooms', 'plant'] + [
    'neighbourhood_Diagonal Mar i el Front Marítim del Poblenou',
    'neighbourhood_OTHER',
    'neighbourhood_Pedralbes',
    'neighbourhood_Sant Antoni',
    'neighbourhood_Sant Gervasi - Galvany',
    'neighbourhood_Sant Gervasi - la Bonanova',
    'neighbourhood_Sant Pere, Santa Caterina i la Ribera',
    'neighbourhood_Sants',
    'neighbourhood_Sarrià',
    'neighbourhood_el Baix Guinardó',
    'neighbourhood_el Barri Gòtic',
    "neighbourhood_el Camp d'en Grassot i Gràcia Nova",
    "neighbourhood_el Camp de l'Arpa del Clot",
    "neighbourhood_el Congrés i els Indians",
    'neighbourhood_el Fort Pienc',
    'neighbourhood_el Guinardó',
    'neighbourhood_el Poble-sec',
    'neighbourhood_el Poblenou',
    'neighbourhood_el Putxet i el Farró',
    'neighbourhood_el Raval',
    "neighbourhood_l'Antiga Esquerra de l'Eixample",
    'neighbourhood_la Barceloneta',
    "neighbourhood_la Dreta de l'Eixample",
    'neighbourhood_la Maternitat i Sant Ramon',
    "neighbourhood_la Nova Esquerra de l'Eixample",
    'neighbourhood_la Sagrada Família',
    'neighbourhood_la Vila Olímpica del Poblenou',
    'neighbourhood_la Vila de Gràcia',
    'neighbourhood_les Corts',
    'neighbourhood_les Tres Torres'] + ['lift_lift', 'parking_yes']

    data = data.reindex(columns=original_columns_order)
    # Make predictions
    predictions = model.predict(data)

    # Display the prediction result
    st.write("Prediction:", predictions)