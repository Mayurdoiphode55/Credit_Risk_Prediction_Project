import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PATHS ---
MODEL_PATH = 'risk_model.joblib'
TRAIN_DATA_PATH = 'train.csv'
TEST_DATA_PATH = 'test.csv'

# --- LOAD SAVED OBJECTS ---
@st.cache_resource
def load_model_pipeline():
    """Loads the pre-trained model pipeline from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at '{MODEL_PATH}'. Please run your notebook to create it.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- LOAD DATA ---
@st.cache_data
def load_data():
    """Loads, combines, and cleans the credit risk datasets."""
    try:
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH)
        test_data['Loan_Status'] = 'Unknown'
        full_data = pd.concat([train_data, test_data], ignore_index=True)

        # --- FIX: Clean the 'Dependents' column to resolve the TypeError ---
        # Fill empty values with the most frequent value (the mode)
        full_data['Dependents'].fillna(full_data['Dependents'].mode()[0], inplace=True)
        # Replace '3+' with just '3' so it can be sorted properly
        full_data['Dependents'] = full_data['Dependents'].replace('3+', '3')

        return full_data
    except FileNotFoundError as e:
        st.error(f"Data file not found. Make sure '{TRAIN_DATA_PATH}' and '{TEST_DATA_PATH}' are in the directory. Error: {e}")
        return None

# --- MAIN APP ---
def main():
    # Load all necessary assets
    model = load_model_pipeline()
    data = load_data()

    # --- SIDEBAR ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Exploratory Data Analysis", "🔮 Make a Prediction"])

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This dashboard uses a Machine Learning model to predict loan approval status based on applicant details."
    )

    # --- HOME PAGE ---
    if page == "🏠 Home":
        st.title("Credit Risk Prediction Dashboard")
        st.markdown("Welcome! Use the navigation panel on the left to explore the data or make a prediction.")
        st.header("Project Overview")
        st.write(
            "This project aims to predict whether a loan application will be approved or rejected. "
            "The model was trained on a dataset of past loan applications."
        )
        if st.checkbox("Show a snippet of the combined raw data"):
            if data is not None:
                st.write(data.head())

    # --- EXPLORATORY DATA ANALYSIS PAGE ---
    elif page == "📊 Exploratory Data Analysis":
        st.title("📊 Exploratory Data Analysis")
        st.write("Explore the relationships between different applicant features and loan status.")

        if data is not None:
            train_data = data[data['Loan_Status'] != 'Unknown'].copy()
            train_data['Loan_Status'] = train_data['Loan_Status'].map({'Y': 1, 'N': 0})

            col1, col2, col3 = st.columns(3)
            approval_rate = train_data['Loan_Status'].mean() * 100
            avg_loan_amnt = train_data['LoanAmount'].mean()
            avg_income = train_data['ApplicantIncome'].mean()

            col1.metric("Overall Approval Rate", f"{approval_rate:.2f}%")
            col2.metric("Average Loan Amount", f"${avg_loan_amnt:,.0f}")
            col3.metric("Average Applicant Income", f"${avg_income:,.0f}")

            st.markdown("---")

            st.subheader("Loan Approval by Different Categories")
            cat_feature = st.selectbox(
                "Select a feature to see its relationship with loan status:",
                ('Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area')
            )

            fig1 = px.histogram(train_data, x=cat_feature, color='Loan_Status', barmode='group',
                                title=f'Loan Status by {cat_feature}',
                                labels={'Loan_Status': 'Loan Status (0: Rejected, 1: Approved)'})
            st.plotly_chart(fig1, use_container_width=True)

    # --- PREDICTION PAGE ---
    elif page == "🔮 Make a Prediction":
        st.title("🔮 Make a Prediction")
        st.write("Enter the applicant's details below to get a loan approval prediction.")

        if model is not None and data is not None:
            with st.form("prediction_form"):
                st.header("Applicant Information")
                col1, col2 = st.columns(2)

                with col1:
                    Gender = st.selectbox("Gender", options=data['Gender'].unique())
                    Married = st.selectbox("Married", options=data['Married'].unique())
                    # This line will now work correctly because 'Dependents' is clean
                    Dependents = st.selectbox("Dependents", options=sorted(data['Dependents'].unique()))
                    Education = st.selectbox("Education", options=data['Education'].unique())
                    Self_Employed = st.selectbox("Self Employed", options=data['Self_Employed'].unique())

                with col2:
                    ApplicantIncome = st.number_input("Applicant Income ($)", min_value=0, value=5000)
                    CoapplicantIncome = st.number_input("Co-applicant Income ($)", min_value=0.0, value=1500.0)
                    LoanAmount = st.number_input("Loan Amount ($)", min_value=0.0, value=150.0)
                    Loan_Amount_Term = st.number_input("Loan Amount Term (months)", min_value=0.0, value=360.0)
                    Property_Area = st.selectbox("Property Area", options=data['Property_Area'].unique())

                st.header("Credit History")
                Credit_History = st.selectbox("Credit History Available?", options=[1.0, 0.0], format_func=lambda x: 'Yes' if x == 1.0 else 'No')

                submit_button = st.form_submit_button(label="Predict Approval Status")

            if submit_button:
                st.write("---")
                st.subheader("Prediction Result")
                try:
                    input_data = pd.DataFrame({
                        'Gender': [Gender],
                        'Married': [Married],
                        'Dependents': [Dependents],
                        'Education': [Education],
                        'Self_Employed': [Self_Employed],
                        'ApplicantIncome': [ApplicantIncome],
                        'CoapplicantIncome': [CoapplicantIncome],
                        'LoanAmount': [LoanAmount],
                        'Loan_Amount_Term': [Loan_Amount_Term],
                        'Credit_History': [Credit_History],
                        'Property_Area': [Property_Area]
                    })

                    prediction = model.predict(input_data)
                    prediction_proba = model.predict_proba(input_data)

                    if prediction[0] == 1:
                        st.success("Prediction: Loan Approved ✅")
                        st.write(f"Probability of Approval: **{prediction_proba[0][1]*100:.2f}%**")
                    else:
                        st.error("Prediction: Loan Rejected ❌")
                        st.write(f"Probability of Rejection: **{prediction_proba[0][0]*100:.2f}%**")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

        else:
            st.warning("Model not loaded. Cannot make predictions.")


if __name__ == "__main__":
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        st.error(f"FATAL: Make sure '{TRAIN_DATA_PATH}' and '{TEST_DATA_PATH}' are in the same directory.")
    else:
        main()
