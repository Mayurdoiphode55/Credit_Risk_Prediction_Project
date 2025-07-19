import os

# Get the absolute path to the current file (dashboard.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import shap
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join(BASE_DIR, 'risk_model.joblib')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'feature_names.pkl')
SHAP_EXPLAINER_PATH = os.path.join(BASE_DIR, 'shap_explainer.joblib')



# --- LOAD AND CREATE ARTIFACTS ---
@st.cache_resource
def load_and_create_artifacts():
    """
    Loads the model pipeline and creates the SHAP explainer.
    This is the robust, final version.
    """
    # Load the model pipeline
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at '{MODEL_PATH}'. Please run your notebook to create it.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    # Load the training data
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=['Loan_Status', 'Loan_ID'], errors='ignore')
    except FileNotFoundError:
        st.error(f"Training data ('{TRAIN_DATA_PATH}') not found. It's needed to create the SHAP explainer.")
        return model, None
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return model, None

    # Create the SHAP explainer
    explainer = None
    try:
        # We create a purely numerical version of the training data for SHAP's background.
        X_train_processed = model.named_steps['preprocessor'].transform(X_train)

        # To prevent errors, we summarize the background data using SHAP's k-means.
        X_train_summary = shap.kmeans(X_train_processed, 10)

        # We create the KernelExplainer using the model's classifier step and the summarized data.
        explainer = shap.KernelExplainer(model.named_steps['classifier'].predict_proba, X_train_summary)

    except Exception as e:
        st.error(f"Error creating SHAP explainer: {e}")

    return model, explainer


# --- LOAD DATA FOR UI ---
@st.cache_data
def load_data_for_ui():
    """Loads and cleans data for populating UI elements."""
    try:
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH)
        test_data['Loan_Status'] = 'Unknown'
        full_data = pd.concat([train_data, test_data], ignore_index=True)

        # Perform comprehensive data cleaning to match the notebook
        full_data['Dependents'].fillna(full_data['Dependents'].mode()[0], inplace=True)
        full_data['Dependents'] = full_data['Dependents'].replace('3+', '3')

        for col in ['Gender', 'Married', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term', 'Education',
                    'Property_Area']:
            if col in full_data.columns:
                full_data[col].fillna(full_data[col].mode()[0], inplace=True)

        if 'LoanAmount' in full_data.columns:
            full_data['LoanAmount'].fillna(full_data['LoanAmount'].median(), inplace=True)

        return full_data
    except FileNotFoundError as e:
        st.error(f"Data file not found. Error: {e}")
        return None


# --- MAIN APP ---
def main():
    model, explainer = load_and_create_artifacts()
    data_ui = load_data_for_ui()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Exploratory Data Analysis", "🔮 Make a Prediction"])
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard uses a Machine Learning model to predict loan approval status.")

    if page == "🏠 Home":
        st.title("Credit Risk Prediction Dashboard")
        st.markdown("Welcome! Use the navigation panel on the left to explore the data or make a prediction.")
        st.header("Project Overview")
        st.write("This project aims to predict whether a loan application will be approved or rejected.")
        if st.checkbox("Show a snippet of the combined raw data"):
            if data_ui is not None:
                st.write(data_ui.head())

    elif page == "📊 Exploratory Data Analysis":
        st.title("📊 Exploratory Data Analysis")
        st.write("Explore the relationships between different applicant features and loan status.")
        if data_ui is not None:
            train_data_for_eda = data_ui[data_ui['Loan_Status'] != 'Unknown'].copy()
            train_data_for_eda.dropna(subset=['Loan_Status'], inplace=True)

            for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term',
                        'Education', 'Property_Area']:
                if col in train_data_for_eda.columns:
                    train_data_for_eda[col].fillna(train_data_for_eda[col].mode()[0], inplace=True)
            if 'LoanAmount' in train_data_for_eda.columns:
                train_data_for_eda['LoanAmount'].fillna(train_data_for_eda['LoanAmount'].median(), inplace=True)

            train_data_for_eda['Loan_Status'] = train_data_for_eda['Loan_Status'].map({'Y': 1, 'N': 0})

            col1, col2, col3 = st.columns(3)
            approval_rate = train_data_for_eda['Loan_Status'].mean() * 100
            avg_loan_amnt = train_data_for_eda['LoanAmount'].mean()
            avg_income = train_data_for_eda['ApplicantIncome'].mean()
            col1.metric("Overall Approval Rate", f"{approval_rate:.2f}%")
            col2.metric("Average Loan Amount", f"${avg_loan_amnt:,.0f}")
            col3.metric("Average Applicant Income", f"${avg_income:,.0f}")
            st.markdown("---")
            st.subheader("Loan Approval by Different Categories")
            cat_feature = st.selectbox("Select a feature:", (
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'))
            fig1 = px.histogram(train_data_for_eda, x=cat_feature, color='Loan_Status', barmode='group',
                                title=f'Loan Status by {cat_feature}',
                                labels={'Loan_Status': 'Loan Status (0: Rejected, 1: Approved)'})
            st.plotly_chart(fig1, use_container_width=True)

    elif page == "🔮 Make a Prediction":
        st.title("🔮 Make a Prediction")
        st.write("Enter the applicant's details below to get a loan approval prediction.")
        if model is not None and explainer is not None and data_ui is not None:
            with st.form("prediction_form"):
                st.header("Applicant Information")
                col1, col2 = st.columns(2)
                with col1:
                    Gender = st.selectbox("Gender", options=data_ui['Gender'].dropna().unique())
                    Married = st.selectbox("Married", options=data_ui['Married'].dropna().unique())
                    Dependents = st.selectbox("Dependents", options=sorted(data_ui['Dependents'].dropna().unique()))
                    Education = st.selectbox("Education", options=data_ui['Education'].dropna().unique())
                    Self_Employed = st.selectbox("Self Employed", options=data_ui['Self_Employed'].dropna().unique())
                with col2:
                    ApplicantIncome = st.number_input("Applicant Income ($)", min_value=0, value=5000)
                    CoapplicantIncome = st.number_input("Co-applicant Income ($)", min_value=0.0, value=1500.0)
                    LoanAmount = st.number_input("Loan Amount ($)", min_value=0.0, value=150.0)
                    Loan_Amount_Term = st.number_input("Loan Amount Term (months)", min_value=0.0, value=360.0)
                    Property_Area = st.selectbox("Property Area", options=data_ui['Property_Area'].dropna().unique())
                st.header("Credit History")
                Credit_History = st.selectbox("Credit History Available?", options=[1.0, 0.0],
                                              format_func=lambda x: 'Yes' if x == 1.0 else 'No')
                submit_button = st.form_submit_button(label="Predict Approval Status")

            if 'prediction' not in st.session_state:
                st.session_state.prediction = None

            if submit_button:
                try:
                    input_data = pd.DataFrame({
                        'Gender': [Gender], 'Married': [Married], 'Dependents': [Dependents],
                        'Education': [Education], 'Self_Employed': [Self_Employed],
                        'ApplicantIncome': [ApplicantIncome], 'CoapplicantIncome': [CoapplicantIncome],
                        'LoanAmount': [LoanAmount], 'Loan_Amount_Term': [Loan_Amount_Term],
                        'Credit_History': [Credit_History], 'Property_Area': [Property_Area]
                    })
                    st.session_state.prediction = model.predict(input_data)[0]
                    st.session_state.prediction_proba = model.predict_proba(input_data)
                    st.session_state.input_data = input_data
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.session_state.prediction = None

            if st.session_state.prediction is not None:
                st.write("---")
                st.subheader("Prediction Result")
                if st.session_state.prediction == 1:
                    st.success("Prediction: Loan Approved ✅")
                    st.write(f"Probability of Approval: **{st.session_state.prediction_proba[0][1] * 100:.2f}%**")
                else:
                    st.error("Prediction: Loan Rejected ❌")
                    st.write(f"Probability of Rejection: **{st.session_state.prediction_proba[0][0] * 100:.2f}%**")

                st.subheader("💡 Prediction Explanation")
                st.write(
                    "The plot below shows the features that pushed the prediction towards 'Approved' (blue) or 'Rejected' (red).")

                # --- THE DEFINITIVE FIX FOR THE HTML PLOT ---
                # 1. Preprocess the input data for the explainer
                input_data_processed = model.named_steps['preprocessor'].transform(st.session_state.input_data)

                # 2. Get the SHAP values
                shap_values = explainer.shap_values(input_data_processed)
                class_to_explain = st.session_state.prediction

                # 3. Generate the plot object
                force_plot = shap.force_plot(
                    explainer.expected_value[class_to_explain],
                    shap_values[class_to_explain],
                    feature_names=model.named_steps['preprocessor'].get_feature_names_out(),
                    matplotlib=False
                )

                # 4. Extract the raw HTML from the object and render it
                shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                st.components.v1.html(shap_html, height=200)

        else:
            st.warning("Model or explainer could not be created. Please check the file paths and logs.")


if __name__ == "__main__":
    main()
