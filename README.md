# Student-Predictor
# =============================================
# AI Student Performance Predictor - PRO VERSION
# Google Colab Ready
# =============================================

# Install dependencies


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =============================================
# STREAMLIT APP
# =============================================
st.title("🎓 AI Student Performance Predictor (PRO)")
st.markdown("Upload your own dataset or use the default one")

# -----------------------------
# SAMPLE DATA (Fallback)
# -----------------------------
default_data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Sleep": [3, 4, 2, 5, 6, 7, 8, 9, 5, 7],
    "Department": ["Science", "Arts", "Science", "Arts", "Science", 
                   "Arts", "Science", "Arts", "Science", "Arts"],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
})

# -----------------------------
# UPLOAD DATA
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file (optional)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except:
        st.error("Error reading file. Please upload a valid CSV.")
        data = default_data
else:
    data = default_data
    st.info("Using default dataset (you can upload your own CSV)")

# Dataset Preview
st.write("### 📊 Dataset Preview")
st.dataframe(data.head(), use_container_width=True)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
if "Hours" in data.columns and "Sleep" in data.columns:
    data["Study_Efficiency"] = data["Hours"] * data["Sleep"]
else:
    st.error("Dataset must contain 'Hours' and 'Sleep' columns!")
    st.stop()

# -----------------------------
# ENCODING
# -----------------------------
if "Department" in data.columns:
    data = pd.get_dummies(data, columns=["Department"], drop_first=True)

# -----------------------------
# PREPARE DATA FOR TRAINING
# -----------------------------
if "Pass" not in data.columns:
    st.error("Target column 'Pass' not found in dataset!")
    st.stop()

X = data.drop("Pass", axis=1)
y = data["Pass"]

# Avoid errors with very small datasets
test_size = 0.3 if len(data) > 10 else 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

st.success(f"### 🎯 Model Trained | Accuracy: **{acc:.2%}**")

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.subheader("🔮 Make a Prediction")

col1, col2 = st.columns(2)
with col1:
    hours = st.number_input("Hours Studied", min_value=1, max_value=20, value=6)
with col2:
    sleep = st.number_input("Sleep Hours", min_value=1, max_value=12, value=6)

dept = st.selectbox("Department", ["Science", "Arts"])

# Prepare input data safely
input_data = pd.DataFrame({
    "Hours": [hours],
    "Sleep": [sleep],
    "Study_Efficiency": [hours * sleep]
})

# Add dummy columns to match training features
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match training data
input_data = input_data[X.columns]

# Predict Button
if st.button("🔮 Predict Student Outcome", type="primary"):
    result = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    if result == 1:
        st.success(f"✅ **Student is likely to PASS**")
        st.info(f"Confidence: **{prob:.1%}**")
    else:
        st.error(f"❌ **Student is likely to FAIL**")
        st.info(f"Confidence: **{(1-prob):.1%}**")

st.caption("Note: This is a demonstration model. Accuracy depends heavily on your dataset.")
