import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Config
st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="centered")

# CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 2rem;
        border-radius: 10px;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        color: gray;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🔬 AI-Powered Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your symptoms and let AI guide you</div>', unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# Load model and labels
if not os.path.exists("disease_prediction_model.pkl") or not os.path.exists("disease_labels.pkl"):
  st.error("Model or label files are missing. Please upload them.")
else:
    model = joblib.load('disease_prediction_model.pkl')
    disease_classes = joblib.load('disease_labels.pkl')

    # Sidebar Inputs
    st.sidebar.header("🧾 Enter Your Details")
    age = st.sidebar.slider("Age", 20, 70, 30)
    fever = st.sidebar.radio("Do you have a fever?", ["No", "Yes"])
    cough = st.sidebar.radio("Do you have a cough?", ["No", "Yes"])
    fatigue = st.sidebar.radio("Do you feel fatigued?", ["No", "Yes"])
    gender = st.sidebar.selectbox("Select Gender", ["Male", "Female"])
    smoker = st.sidebar.selectbox("Are you a smoker?", ["No", "Yes"])

    # Preprocessing Function
    def preprocess_input(age, fever, cough, fatigue, gender, smoker):
        gender_map = {'Male': 0, 'Female': 1}
        smoker_map = {'No': 0, 'Yes': 1}
        input_data = pd.DataFrame([[age, int(fever == "Yes"), int(cough == "Yes"), int(fatigue == "Yes"), gender_map[gender], smoker_map[smoker]]], 
                                  columns=['Age', 'Fever', 'Cough', 'Fatigue', 'Gender', 'Smoker'])
        return input_data.astype(float)

    # Predict Button
    if st.sidebar.button("🔮 Predict Disease"):
        st.info("Analyzing your symptoms...")
        input_data = preprocess_input(age, fever, cough, fatigue, gender, smoker)

        with st.spinner("Processing..."):
            try:
                prediction = model.predict(input_data)[0]
                predicted_disease = disease_classes[prediction]
                st.success(f"🩺 AI Suggests: *{predicted_disease}*")
                st.balloons()

                # Display symptom summary
                with st.expander("📄 View Symptom Summary"):
                    st.table(input_data.rename(columns={
                        "Age": "Age",
                        "Fever": "Fever (1=Yes)",
                        "Cough": "Cough (1=Yes)",
                        "Fatigue": "Fatigue (1=Yes)",
                        "Gender": "Gender (0=Male, 1=Female)",
                        "Smoker": "Smoker (1=Yes)"
                    }))
            except Exception as e:
                st.error("Prediction failed. Please check input or model. " + str(e))

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="footer">This AI tool is for informational purposes only. Consult a doctor for medical advice.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
    <style>
    .main {
        background-image: url('https://www.istockphoto.com/vector/vector-set-of-design-templates-and-elements-for-healthcare-and-medicine-in-trendy-gm1125924208-296187989.jpg');
        background-size: cover;
        background-position: center;
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)
st.image("https://www.freepik.com/free-photos-vectors/medical-banner?log-in=google.jpg", use_column_width=True)
st.markdown("""
    <style>
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        animation: fadeIn 2s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔬 AI-Powered Disease Prediction</div>', unsafe_allow_html=True)
import plotly.express as px

df = pd.DataFrame({
    "Symptom": ["Fever", "Cough", "Fatigue"],
    "Cases": [100, 250, 180]
})

fig = px.bar(df, x="Symptom", y="Cases", title="Symptom Distribution", animation_frame="Symptom")
st.plotly_chart(fig)
st.image("https://lottiefiles.com/free-animation/heartbeat-medical-pPbWnDhphP.gif", width=200)