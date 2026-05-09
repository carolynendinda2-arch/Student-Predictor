import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =============================
# TITLE
# =============================
st.title("🎓 AI Student Performance Predictor (PRO)")
st.markdown("Upload your dataset or use default data")

# =============================
# DEFAULT DATA
# =============================
default_data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Sleep": [3, 4, 2, 5, 6, 7, 8, 9, 5, 7],
    "Department": ["Science", "Arts", "Science", "Arts", "Science",
                   "Arts", "Science", "Arts", "Science", "Arts"],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
})

# =============================
# UPLOAD DATA
# =============================
uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    data = default_data
    st.info("Using default dataset")

# =============================
# DATA PREVIEW
# =============================
st.write("### Dataset Preview")
st.dataframe(data)

# =============================
# FEATURE ENGINEERING
# =============================
data["Study_Efficiency"] = data["Hours"] * data["Sleep"]

# ENCODING
data = pd.get_dummies(data, columns=["Department"])

# =============================
# SPLIT DATA
# =============================
X = data.drop("Pass", axis=1)
y = data["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# MODEL
# =============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

st.success(f"Model Accuracy: {acc:.2f}")

# =============================
# INPUT SECTION
# =============================
st.subheader("Make Prediction")

hours = st.number_input("Hours Studied", 1, 20, 5)
sleep = st.number_input("Sleep Hours", 1, 12, 6)
dept = st.selectbox("Department", ["Science", "Arts"])

# =============================
# INPUT DATA
# =============================
input_data = pd.DataFrame({
    "Hours": [hours],
    "Sleep": [sleep],
    "Study_Efficiency": [hours * sleep],
    "Department_Arts": [1 if dept == "Arts" else 0],
    "Department_Science": [1 if dept == "Science" else 0]
})

# MATCH COLUMNS
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X.columns]

# =============================
# PREDICTION
# =============================
if st.button("Predict"):
    result = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if result == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")

    st.info(f"Confidence: {prob:.2%}")

st.caption("Demo project — results depend on dataset quality")
