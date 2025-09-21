import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PATHS ---
MODEL_PATH = 'risk_model.joblib'

# --- DATABASE CONNECTION DETAILS ---
DB_CONFIG = {
    "drivername": "mysql+pymysql",
    "username": "root",
    "password": "Mayur9730@",
    "host": "127.0.0.1",
    "database": "credit_risk_db"
}

# --- HELPER FUNCTION TO SAVE PREDICTIONS ---
def save_prediction_to_db(engine, application_id, status, prob, shap_values_dict):
    """
    Saves a prediction and its SHAP explanations to the database.
    """
    try:
        with engine.begin() as conn:
            insert_prediction_query = text(
                "INSERT INTO Predictions(Application_Id, Predicted_Status, Probability, Prediction_Date) "
                "VALUES (:app_id, :status, :prob, NOW())"
            )
            result = conn.execute(
                insert_prediction_query,
                {"app_id": application_id, "status": status, "prob": prob}
            )
            prediction_id = result.lastrowid

            insert_explanation_query = text(
                "INSERT INTO Explanations(Prediction_Id, Feature_Name, Shap_Value) "
                "VALUES (:pred_id, :feat_name, :shap_val)"
            )
            for feature, shap_value in shap_values_dict.items():
                conn.execute(
                    insert_explanation_query,
                    {"pred_id": prediction_id, "feat_name": feature, "shap_val": shap_value}
                )
        st.sidebar.success(f"Prediction for Application ID {application_id} saved to database!")
    except Exception as e:
        st.sidebar.error(f"Database Error: Failed to save prediction. {e}")


# --- LOAD AND CREATE ARTIFACTS (FROM DB) ---
@st.cache_resource
def load_and_create_artifacts():
    """
    Loads the model pipeline from disk and creates the SHAP explainer
    by fetching training data from the MySQL database.
    """
    model = None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

    X_train = None
    try:
        connection_url = URL.create(**DB_CONFIG)
        engine = create_engine(connection_url)
        sql_query = """
        SELECT 
        c.gender AS Gender,
        c.married AS Married,
        c.dependents AS Dependents,
        c.education AS Education,
        c.self_employed AS Self_Employed,
        c.income AS ApplicantIncome,
        0.0 AS CoapplicantIncome,
        la.LoanAmount AS LoanAmount,
        la.Loan_Amount_Term AS Loan_Amount_Term,
        la.Credit_History AS Credit_History,
        la.Property_Area AS Property_Area,
        la.Status AS Loan_Status
        FROM Loan_Applications la
        JOIN Customers c 
        ON la.customer_id = c.customer_id;
        """
        train_df = pd.read_sql(sql_query, engine)

        X_train = train_df.rename(columns={
            'gender': 'Gender',
            'married': 'Married',
            'dependents': 'Dependents',
            'education': 'Education',
            'self_employed': 'Self_Employed'
        })
        X_train = X_train.drop(columns=['Loan_Status'], errors='ignore')
    except Exception as e:
        st.error(f"Failed to connect to the database or load data for SHAP: {e}")
        return None, None
    
    explainer = None
    if X_train is not None:
        try:
            X_train_processed = model.named_steps['preprocessor'].transform(X_train)
            X_train_summary = shap.kmeans(X_train_processed, 10) 
            explainer = shap.KernelExplainer(model.named_steps['classifier'].predict_proba, X_train_summary)
        except Exception as e:
            st.error(f"An error occurred while creating the SHAP explainer: {e}")
            return model, None

    return model, explainer


# --- LOAD DATA FOR UI (FROM DB) ---
@st.cache_data
def load_data_for_ui():
    """
    Loads data from the MySQL database to populate UI elements and for EDA.
    """
    try:
        connection_url = URL.create(**DB_CONFIG)
        engine = create_engine(connection_url)
        sql_query = """
        SELECT 
        c.gender AS Gender,
        c.married AS Married,
        c.dependents AS Dependents,
        c.education AS Education,
        c.self_employed AS Self_Employed,
        c.income AS ApplicantIncome,
        0.0 AS CoapplicantIncome,
        la.LoanAmount AS LoanAmount,
        la.Loan_Amount_Term AS Loan_Amount_Term,
        la.Credit_History AS Credit_History,
        la.Property_Area AS Property_Area,
        la.Status AS Loan_Status
        FROM Loan_Applications la
        JOIN Customers c 
        ON la.customer_id = c.customer_id;
        """
        full_data = pd.read_sql(sql_query, engine)
        
        # --- Data Cleaning ---
        full_data['Dependents'].fillna(full_data['Dependents'].mode()[0], inplace=True)
        if full_data['Dependents'].dtype == 'object':
            full_data['Dependents'] = full_data['Dependents'].replace('3+', '3')

        for col in ['Gender', 'Married', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term', 
                    'Education', 'Property_Area']:
            if col in full_data.columns:
                full_data[col].fillna(full_data[col].mode()[0], inplace=True)
        if 'LoanAmount' in full_data.columns:
            full_data['LoanAmount'].fillna(full_data['LoanAmount'].median(), inplace=True)
        return full_data
    except Exception as e:
        st.error(f"UI Data Error: Could not load data from the database. {e}")
        return None


# --- MAIN APP ---
def main():
    model, explainer = load_and_create_artifacts()
    data_ui = load_data_for_ui()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📊 Exploratory Data Analysis", "🔮 Make a Prediction", "📈 Prediction History", "🌍 Global Feature Importance"]
    )
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard uses a Machine Learning model to predict loan approval status.")

    if page == "🏠 Home":
        st.title("Credit Risk Prediction Dashboard")
        st.markdown("Welcome! Use the navigation panel on the left to explore the data or make a prediction.")
        st.header("Project Overview")
        st.write("This project aims to predict whether a loan application will be approved or rejected.")
        if st.checkbox("Show a snippet of the raw data from the database"):
            if data_ui is not None:
                st.write(data_ui.head())

    elif page == "📊 Exploratory Data Analysis":
        st.title("📊 Exploratory Data Analysis")
        st.write("Explore the relationships between different applicant features and loan status.")
        if data_ui is not None:
            train_data_for_eda = data_ui.copy()
            train_data_for_eda.dropna(subset=['Loan_Status'], inplace=True)
            train_data_for_eda['Loan_Status_Binary'] = train_data_for_eda['Loan_Status'].map({'Y': 1, 'N': 0})
            col1, col2, col3 = st.columns(3)
            approval_rate = train_data_for_eda['Loan_Status_Binary'].mean() * 100
            avg_loan_amnt = train_data_for_eda['LoanAmount'].mean()
            avg_income = train_data_for_eda['ApplicantIncome'].mean()
            col1.metric("Overall Approval Rate", f"{approval_rate:.2f}%")
            col2.metric("Average Loan Amount", f"${avg_loan_amnt:,.0f}")
            col3.metric("Average Applicant Income", f"${avg_income:,.0f}")
            st.markdown("---")
            st.subheader("Loan Approval by Different Categories")
            cat_feature = st.selectbox("Select a feature:", ('Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'))
            fig1 = px.histogram(train_data_for_eda, x=cat_feature, color='Loan_Status', barmode='group',
                                title=f'Loan Status by {cat_feature}',
                                labels={'Loan_Status': 'Loan Status'})
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

            if submit_button:
                try:
                    input_data = pd.DataFrame({'Gender': [Gender], 'Married': [Married], 'Dependents': [Dependents], 'Education': [Education], 'Self_Employed': [Self_Employed], 'ApplicantIncome': [ApplicantIncome], 'CoapplicantIncome': [CoapplicantIncome], 'LoanAmount': [LoanAmount], 'Loan_Amount_Term': [Loan_Amount_Term], 'Credit_History': [Credit_History], 'Property_Area': [Property_Area]})
                    st.session_state.prediction = model.predict(input_data)[0]
                    st.session_state.prediction_proba = model.predict_proba(input_data)
                    st.session_state.input_data = input_data
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.session_state.prediction = None

            # --- SHAP Explanation + Save (robust) ---
            if st.session_state.get('prediction') is not None:
                st.write("---")
                st.subheader("💡 Prediction Explanation")
                st.write("The plot shows features pushing the prediction towards 'Approved' (blue) or 'Rejected' (red).")

                try:
                    # --- Preprocess the input ---
                    input_data = st.session_state.input_data
                    input_data_processed = model.named_steps['preprocessor'].transform(input_data)

                    # --- Get feature names from preprocessor (expanded after encoding) ---
                    feature_names = model.named_steps['preprocessor'].get_feature_names_out()

                    # --- Get SHAP values ---
                    shap_values_raw = explainer.shap_values(input_data_processed)

                    # --- Select correct class index ---
                    class_arr = np.array(model.named_steps['classifier'].classes_, dtype=object)
                    pred_class = st.session_state.prediction
                    matches = np.where(class_arr == pred_class)[0]
                    class_index = int(matches[0]) if matches.size > 0 else 0

                    # --- Extract SHAP values for the predicted class ---
                    if isinstance(shap_values_raw, list):
                        shap_values_arr = np.array(shap_values_raw[class_index])
                    else:
                        shap_values_arr = np.array(shap_values_raw)

                    # For a single sample → ensure 1D
                    if shap_values_arr.ndim == 2:
                        sample_shap = shap_values_arr[0]
                    else:
                        sample_shap = shap_values_arr.reshape(-1)

                    # --- Align SHAP values with features ---
                    sample_shap = sample_shap[:len(feature_names)]
                    features_series = pd.Series(input_data_processed.ravel(), index=feature_names[:len(sample_shap)])

                    # --- Base value ---
                    expected_val = explainer.expected_value
                    if isinstance(expected_val, (list, np.ndarray)):
                        base_value = float(np.array(expected_val).ravel()[class_index])
                    else:
                        base_value = float(expected_val)

                    # --- SHAP force plot ---
                    force_obj = shap.force_plot(base_value, sample_shap, features_series, matplotlib=False)
                    st.components.v1.html(f"<head>{shap.getjs()}</head><body>{force_obj.html()}</body>", height=420)

                except Exception as e:
                    st.error(f"SHAP explanation could not be generated: {e}")


                # --- Save Prediction to Database ---
                if st.button("Save Prediction to Database"):
                    try:
                        # determine probability for the predicted class
                        try:
                            prob_index = class_index
                            probability = float(st.session_state.prediction_proba[0][prob_index])
                        except Exception:
                            # fallback: probability of predicted class if mapping fails
                            probability = float(np.max(st.session_state.prediction_proba[0]))

                        # determine human label: prefer numeric 1 => Approved else use probability threshold
                        try:
                            pred_status = "Approved" if (isinstance(pred_class, (int, np.integer)) and int(pred_class) == 1) or (probability >= 0.5) else "Rejected"
                        except Exception:
                            pred_status = "Approved" if probability >= 0.5 else "Rejected"

                        placeholder_app_id = 1

                        # ensure shap and feature alignment, truncate if necessary
                        usable_feature_names = list(feature_names[:len(sample_shap)])
                        shap_dict = dict(zip(usable_feature_names, np.array(sample_shap).flatten()))

                        connection_url = URL.create(**DB_CONFIG)
                        engine = create_engine(connection_url)
                        save_prediction_to_db(engine, placeholder_app_id, pred_status, probability, shap_dict)

                    except Exception as e:
                        st.error(f"Could not save prediction: {e}")

    elif page == "📈 Prediction History":
        st.title("📈 Prediction History & Monitoring")
        st.write("This page displays the history of all predictions made and saved to the database.")
        try:
            connection_url = URL.create(**DB_CONFIG)
            engine = create_engine(connection_url)
            query = """
            SELECT 
                p.Prediction_Id,
                p.Application_Id,
                p.Predicted_Status,
                p.Probability,
                p.Prediction_Date
            FROM Predictions p
            ORDER BY p.Prediction_Date DESC;
            """
            history_df = pd.read_sql(query, engine)
            if not history_df.empty:
                st.subheader("Key Metrics")
                total_predictions = len(history_df)
                approval_rate = (history_df['Predicted_Status'].value_counts().get('Approved', 0) / total_predictions) * 100
                col1, col2 = st.columns(2)
                col1.metric("Total Predictions Made", f"{total_predictions}")
                col2.metric("Overall Approval Rate", f"{approval_rate:.2f}%")
                st.markdown("---")
                st.subheader("Prediction Distribution")
                fig = px.pie(history_df, names='Predicted_Status', title='Distribution of Approved vs. Rejected Predictions', color='Predicted_Status', color_discrete_map={'Approved':'#00B09B', 'Rejected':'#F5567B'})
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                st.subheader("Recent Predictions")
                history_df['Probability'] = history_df['Probability'].apply(lambda x: f"{x*100:.2f}%")
                history_df['Prediction_Date'] = pd.to_datetime(history_df['Prediction_Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(history_df.head(20), use_container_width=True)
            else:
                st.warning("No predictions found in the database yet.")
        except Exception as e:
            st.error(f"An error occurred while fetching prediction history: {e}")

    elif page == "🌍 Global Feature Importance":
        st.title("🌍 Global Feature Importance")
        st.write("This page analyzes all saved SHAP explanations to show which features have the most impact on the model's predictions overall.")
        st.info("This is calculated by taking the average of the absolute SHAP values for each feature across all predictions.")
        try:
            connection_url = URL.create(**DB_CONFIG)
            engine = create_engine(connection_url)
            query = """
            SELECT 
                Feature_Name,
                AVG(ABS(Shap_Value)) AS mean_abs_shap_value
            FROM Explanations
            GROUP BY Feature_Name
            ORDER BY mean_abs_shap_value DESC;
            """
            feature_importance_df = pd.read_sql(query, engine)
            if not feature_importance_df.empty:
                st.subheader("Overall Feature Impact on Model Output")
                fig = px.bar(
                    feature_importance_df,
                    x='mean_abs_shap_value',
                    y='Feature_Name',
                    orientation='h',
                    title='Global Feature Importance',
                    labels={'mean_abs_shap_value': 'Mean Absolute SHAP Value (Impact)', 'Feature_Name': 'Feature'}
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No explanation data found. Please make and save a few predictions first.")
        except Exception as e:
            st.error(f"An error occurred while calculating feature importance: {e}")

if __name__ == "__main__":
    main()
