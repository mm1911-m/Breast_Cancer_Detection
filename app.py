import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Detection", page_icon="🔬", layout="wide")

@st.cache_resource
def load_and_train_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, data, accuracy

model, data, accuracy = load_and_train_model()

st.sidebar.header("📝 Patient Data Input")
st.sidebar.markdown("Adjust the sliders to enter measurement values:")

feature_names = data.feature_names
min_vals = np.min(data.data, axis=0)
max_vals = np.max(data.data, axis=0)
default_vals = data.data[0]
user_inputs = {}

groups = {
    'Radius Features': [0, 10, 20],
    'Texture Features': [1, 11, 21],
    'Perimeter Features': [2, 12, 22],
    'Area Features': [3, 13, 23],
    'Smoothness Features': [4, 14, 24],
    'Compactness Features': [5, 15, 25],
    'Concavity Features': [6, 16, 26],
    'Concave Points Features': [7, 17, 27],
    'Symmetry Features': [8, 18, 28],
    'Fractal Dimension Features': [9, 19, 29]
}

for group_name, indices in groups.items():
    with st.sidebar.expander(group_name, expanded=False):
        for i in indices:
            clean_name = feature_names[i].replace('_mean', '').replace('_se', '').replace('_worst', '').capitalize()
            val = st.slider(
                f"{clean_name} ({feature_names[i]})",
                min_value=float(min_vals[i]*0.8),
                max_value=float(max_vals[i]*1.2),
                value=float(default_vals[i]),
                step=0.01,
                key=feature_names[i]
            )
            user_inputs[i] = val

sorted_input = [user_inputs[i] for i in range(30)]
patient_array = np.array(sorted_input).reshape(1, -1)

st.title("🔬 Breast Cancer Detection System (ML Demo)")
st.markdown(f"""
This application uses a **Random Forest Classifier** to predict tumor type based on measurements.
**Model Accuracy:** `{accuracy*100:.1f}%`
""")

st.write("---")

if st.button("🚀 Analyze & Predict", type="primary"):
    with st.spinner('Analyzing data...'):
        prediction = model.predict(patient_array)[0]
        probabilities = model.predict_proba(patient_array)[0]
        malignant_prob = probabilities[0]
        benign_prob = probabilities[1]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Diagnosis Result")
        if prediction == 0:
            st.error(f"🚨 **Result: Malignant Tumor**")
            st.markdown(f"Malignancy Probability: **{malignant_prob*100:.1f}%**")
        else:
            st.success(f"✅ **Result: Benign Tumor**")
            st.markdown(f"Benign Probability: **{benign_prob*100:.1f}%**")
        st.info("Note: This result is based on an AI model and does not replace professional medical advice.")

    with col2:
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots()
        colors = ['#ff4b4b', '#4caf50']
        ax.pie([malignant_prob, benign_prob], labels=['Malignant', 'Benign'], autopct='%1.1f%%', colors=colors, startangle=90)
        ax.axis('equal') 
        st.pyplot(fig)

    st.write("---")
    st.subheader("📈 Key Factors Influencing Prediction")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-5:]
    
    fig_bar, ax_bar = plt.subplots(figsize=(10, 3))
    ax_bar.barh(range(len(indices)), importances[indices], color='#3498db', align='center')
    ax_bar.set_yticks(range(len(indices)))
    ax_bar.set_yticklabels([data.feature_names[i] for i in indices])
    ax_bar.set_xlabel('Relative Importance')
    st.pyplot(fig_bar)

else:
    st.info("👈 Please enter patient data from the sidebar and click the button to start analysis.")