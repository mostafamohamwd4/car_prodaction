import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load the trained model
# -----------------------------
with open('./Model/BestModel.pkl', 'rb') as file:  # replace with your model filename
    model = pickle.load(file)

st.title("Car Price Prediction App")

st.write("""
This app predicts the car price based on input features.
""")

# -----------------------------
# User input for features
# -----------------------------
def user_input_features():
    selling_price = st.number_input("Selling Price (e.g., 450000)", value=200000)
    km_driven = st.number_input("Kilometers Driven (e.g., 145500)", value=100000)
    
    fuel = st.selectbox("Fuel Type", options=[1, 2, 3], help="1: Petrol, 2: Diesel, 3: CNG")
    seller_type = st.selectbox("Seller Type", options=[1, 2, 3], help="1: Individual, 2: Dealer, 3: Trustmark Dealer")
    transmission = st.selectbox("Transmission", options=[1, 2], help="1: Manual, 2: Automatic")
    owner = st.number_input("Owner Type (0,1,2,3...)", value=0)
    
    mileage = st.text_input("Mileage (e.g., 23.4 kmpl)", value="20 kmpl")
    engine = st.text_input("Engine (e.g., 1248 CC)", value="1200 CC")
    max_power = st.text_input("Max Power (e.g., 74 bhp)", value="75 bhp")
    seats = st.number_input("Seats (e.g., 5)", value=5)
    age = st.number_input("Car Age (years)", value=5)

    data = {
        "selling_price": selling_price,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "age": age
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    # Optional: Preprocessing if your model needs numerical values from strings
    # e.g., convert "23.4 kmpl" -> 23.4
    input_df['mileage'] = input_df['mileage'].str.replace(' kmpl','').astype(float)
    input_df['engine'] = input_df['engine'].str.replace(' CC','').astype(float)
    input_df['max_power'] = input_df['max_power'].str.replace(' bhp','').astype(float)

    prediction = model.predict(input_df)
    st.success(f"The predicted car price is: {prediction[0]:,.2f}")
