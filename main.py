import streamlit as st
import pickle
import numpy as np

# Step 1: Load your trained model
with open("\insurance_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Medical Insurance Cost Prediction App 💰🏥")

# Step 2: Take user input
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sex = st.selectbox("Sex", ("male", "female"))
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ("yes", "no"))
region = st.selectbox("Region", ("northwest", "northeast", "southeast", "southwest"))

# Step 3: Convert categorical to numeric (same as training preprocessing)
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

# One-hot encoding for region
region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0
# (northeast is default when all 0)

# Step 4: Prepare input
input_data = np.array([[age, sex, bmi, children, smoker,
                        region_northwest, region_southeast, region_southwest]])

# Step 5: Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Insurance Charges: ${prediction[0]:.2f}")




