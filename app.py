import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_shape=(11,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.load_weights("models/customer_churn.weights.h5")
scaler = joblib.load("models/scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will churn or stay.")

# Inputs
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    value=50000.0
)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    value=100000.0
)

# Prediction button
if st.button("Predict"):

    # Encode gender
    gender = 1 if gender == "Male" else 0

    # Encode geography
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    # Arrange features
    features = np.array([[
        credit_score,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_cr_card,
        is_active_member,
        estimated_salary,
        geo_germany,
        geo_spain
    ]])

    # Scale features
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)

    probability = prediction[0][0]

    st.subheader("Prediction Result")

    if probability > 0.5:
        st.error(
            f"⚠ Customer is likely to CHURN\n\nProbability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Customer is likely to STAY\n\nProbability: {(1-probability):.2%}"
        )

    st.write(f"Raw churn probability: **{probability:.4f}**")