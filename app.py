import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Salary Prediction using years",
    page_icon="💰",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("💼 Salary Prediction App")
st.write("Predict salary based on years of experience")

# ---------------- LOAD MODEL ----------------
model = joblib.load("final_model_SLR.pkl")

# ---------------- INPUT ----------------
years_exp = st.slider(
    "📊 Years of Experience",
    min_value=0.0,
    max_value=30.0,
    step=0.5
)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Salary"):
    input_data = np.array([years_exp]).reshape(-1, 1)
    predicted_salary = model.predict(input_data)

    salary = float(predicted_salary.ravel()[0])

    st.success("✅ Prediction Successful")
    st.metric(
        label="Predicted Salary",
        value=f"₹ {salary:,.2f}"
    )


    

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Salary Prediction using Machine Learning & Streamlit")