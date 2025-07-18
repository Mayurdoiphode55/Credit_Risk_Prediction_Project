pip install streamlit

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib  # Or use pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PATHS ---
# IMPORTANT: Make sure these paths are correct.
# You should first run your notebook to train and save the model and preprocessor.
MODEL_PATH = 'risk_model.joblib'
PREPROCESSOR_PATH = 'preprocessor.joblib'
DATA_PATH = 'credit_risk_dataset.csv'

# --- LOAD SAVED OBJECTS ---
# This function will load your trained model and preprocessor.
# It uses caching to avoid reloading on every interaction.
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and preprocessor from disk."""
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at '{MODEL_PATH}'. Please train and save your model first.")
        model = None
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except FileNotFoundError:
        st.error(f"Preprocessor file not found at '{PREPROCESSOR_PATH}'. Please train and save your preprocessor first.")
        preprocessor = None
    return model, preprocessor

# --- LOAD DATA ---
# This function loads the dataset for the EDA section.
@st.cache_data
def load_data():
    """Loads the credit risk dataset."""
    try:
        data = pd.read_csv(DATA_PATH)
        # Basic data cleaning from your notebook
        data.dropna(inplace=True)
        return data
    except FileNotFoundError:
        st.error(f"Data file not found at '{DATA_PATH}'. Make sure the CSV is in the same directory.")
        return None

# --- MAIN APP ---
def main():
    # Load all necessary assets
    model, preprocessor = load_assets()
    data = load_data()

    # --- SIDEBAR ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Exploratory Data Analysis", "🔮 Make a Prediction"])

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This is a web app created to provide insights from a credit risk dataset "
        "and predict loan default probability using a Machine Learning model."
    )

    # --- HOME PAGE ---
    if page == "🏠 Home":
        st.title("Credit Risk Prediction Dashboard")
        st.markdown("Welcome to the dashboard. Use the navigation panel on the left to explore the data or make a prediction.")

        st.header("Project Overview")
        st.write(
            "The goal of this project is to predict whether a borrower will default on a loan. "
            "This dashboard provides two main functionalities:"
        )
        st.markdown("""
            * **Exploratory Data Analysis:** Dive deep into the dataset with interactive visualizations to understand the factors that influence credit risk.
            * **Make a Prediction:** Use the trained machine learning model to get a real-time risk assessment for a new loan applicant.
        """)

        if st.checkbox("Show a snippet of the raw data"):
            if data is not None:
                st.write(data.head())
            else:
                st.warning("Could not display data. Please check data file path.")

    # --- EXPLORATORY DATA ANALYSIS PAGE ---
    elif page == "📊 Exploratory Data Analysis":
        st.title("📊 Exploratory Data Analysis")
        st.write("Explore the relationships between different features and the loan status.")

        if data is not None:
            # --- Key Metrics ---
            col1, col2, col3 = st.columns(3)
            default_rate = data['loan_status'].mean() * 100
            avg_loan_amnt = data['loan_amnt'].mean()
            avg_income = data['person_income'].mean()

            col1.metric("Overall Default Rate", f"{default_rate:.2f}%")
            col2.metric("Average Loan Amount", f"${avg_loan_amnt:,.0f}")
            col3.metric("Average Applicant Income", f"${avg_income:,.0f}")

            st.markdown("---")

            # --- Interactive Charts ---
            st.subheader("Loan Status by Different Categories")
            # Create a dropdown to select a categorical feature
            cat_feature = st.selectbox(
                "Select a feature to see its relationship with loan status:",
                ('person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file')
            )

            # Bar chart showing default rate by the selected category
            fig1 = px.histogram(data, x=cat_feature, color='loan_status', barmode='group',
                                title=f'Loan Status by {cat_feature.replace("_", " ").title()}',
                                labels={'loan_status': 'Loan Status (0: Non-Default, 1: Default)'})
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Exploring Numerical Features")
            # Scatter plot
            fig2 = px.scatter(data, x='person_income', y='loan_amnt', color='loan_status',
                              title='Income vs. Loan Amount',
                              labels={'person_income': 'Applicant Income', 'loan_amnt': 'Loan Amount'},
                              hover_data=['person_age', 'loan_intent'])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Cannot perform analysis because the data could not be loaded.")


    # --- PREDICTION PAGE ---
    elif page == "🔮 Make a Prediction":
        st.title("🔮 Make a Prediction")
        st.write("Enter the applicant's details below to get a credit risk prediction.")

        if model is not None and preprocessor is not None:
            # --- Input Form ---
            with st.form("prediction_form"):
                st.header("Applicant Information")
                col1, col2 = st.columns(2)

                with col1:
                    person_age = st.number_input("Age", min_value=18, max_value=100, value=25)
                    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
                    person_home_ownership = st.selectbox("Home Ownership", options=data['person_home_ownership'].unique())
                    person_emp_length = st.slider("Employment Length (years)", min_value=0, max_value=50, value=5)

                with col2:
                    loan_intent = st.selectbox("Loan Intent", options=data['loan_intent'].unique())
                    loan_grade = st.selectbox("Loan Grade", options=sorted(data['loan_grade'].unique()))
                    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
                    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, format="%.2f")

                st.header("Credit History")
                col3, col4 = st.columns(2)
                with col3:
                    cb_person_default_on_file = st.radio("Has Defaulted Before?", options=['Y', 'N'])
                with col4:
                    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=40, value=3)

                # Submit button
                submit_button = st.form_submit_button(label="Predict Risk")

            if submit_button:
                # --- Create a dataframe from inputs ---
                input_data = pd.DataFrame({
                    'person_age': [person_age],
                    'person_income': [person_income],
                    'person_home_ownership': [person_home_ownership],
                    'person_emp_length': [person_emp_length],
                    'loan_intent': [loan_intent],
                    'loan_grade': [loan_grade],
                    'loan_amnt': [loan_amnt],
                    'loan_int_rate': [loan_int_rate],
                    'loan_percent_income': [(loan_amnt / person_income) if person_income > 0 else 0],
                    'cb_person_default_on_file': [cb_person_default_on_file],
                    'cb_person_cred_hist_length': [cb_person_cred_hist_length]
                })

                st.write("---")
                st.subheader("Prediction Result")

                # --- Preprocess the input data and make prediction ---
                try:
                    # Ensure the column order matches the preprocessor's expectation
                    # This is crucial!
                    input_processed = preprocessor.transform(input_data)
                    prediction = model.predict(input_processed)
                    prediction_proba = model.predict_proba(input_processed)

                    # --- Display the result ---
                    if prediction[0] == 1:
                        st.error("Prediction: High Risk (Likely to Default)")
                    else:
                        st.success("Prediction: Low Risk (Unlikely to Default)")

                    st.write(f"Probability of Default: **{prediction_proba[0][1]*100:.2f}%**")
                    st.write(f"Probability of Non-Default: **{prediction_proba[0][0]*100:.2f}%**")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.warning("Please ensure your saved preprocessor can handle the input data format.")

        else:
            st.warning("Model or preprocessor not loaded. Cannot make predictions.")


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        st.error(f"FATAL: Dataset '{DATA_PATH}' not found. Please download it and place it in the same directory as this script.")
    else:
        main()