import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

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

# Global variables
MODEL = None
EXPLAINER = None
DATA_UI = None
LAST_PREDICTION_DATA = {}
BACKGROUND_DATA = None

def get_db_engine():
    """Creates a SQLAlchemy engine instance."""
    try:
        connection_url = URL.create(**DB_CONFIG)
        return create_engine(connection_url)
    except Exception as e:
        app.logger.error(f"Failed to create database engine: {e}")
        return None

def save_prediction_to_db(application_id, status, prob, shap_values_dict):
    """Saves a prediction and its SHAP explanations to the database."""
    engine = get_db_engine()
    if not engine:
        return False, "Database connection failed."

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
                    {"pred_id": prediction_id, "feat_name": feature, "shap_val": float(shap_value)}
                )
        return True, f"Prediction for Application ID {application_id} saved successfully!"
    except Exception as e:
        app.logger.error(f"Database Error: {e}")
        return False, f"Database Error: {e}"

def load_data_for_ui(engine):
    """Loads and cleans data from the database."""
    sql_query = """
    SELECT 
    c.gender AS Gender, c.married AS Married, c.dependents AS Dependents, 
    c.education AS Education, c.self_employed AS Self_Employed, 
    c.income AS ApplicantIncome, 0.0 AS CoapplicantIncome, la.LoanAmount AS LoanAmount,
    la.Loan_Amount_Term AS Loan_Amount_Term, la.Credit_History AS Credit_History, 
    la.Property_Area AS Property_Area, la.Status AS Loan_Status
    FROM Loan_Applications la
    JOIN Customers c ON la.customer_id = c.customer_id;
    """
    full_data = pd.read_sql(sql_query, engine)
    
    full_data['Dependents'] = full_data['Dependents'].fillna(full_data['Dependents'].mode()[0])
    
    if full_data['Dependents'].dtype == 'object':
        full_data['Dependents'] = full_data['Dependents'].astype(str).replace('3\+', '3', regex=True)

    for col in ['Gender', 'Married', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term', 
                'Education', 'Property_Area']:
        if col in full_data.columns:
            full_data[col] = full_data[col].fillna(full_data[col].mode()[0])
            
    if 'LoanAmount' in full_data.columns:
        full_data['LoanAmount'] = full_data['LoanAmount'].fillna(full_data['LoanAmount'].median())
    
    return full_data

def load_and_create_artifacts():
    """Loads the model and creates the SHAP explainer."""
    global MODEL, EXPLAINER, DATA_UI, BACKGROUND_DATA

    if not os.path.exists(MODEL_PATH):
        app.logger.error(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
        return

    try:
        MODEL = joblib.load(MODEL_PATH)
        app.logger.info("Model loaded successfully")
    except Exception as e:
        app.logger.error(f"FATAL ERROR: Failed to load model: {e}")
        return

    engine = get_db_engine()
    if not engine:
        return
        
    try:
        DATA_UI = load_data_for_ui(engine)
        X_train = DATA_UI.drop(columns=['Loan_Status'], errors='ignore').copy()
        
        app.logger.info("Creating SHAP explainer...")
        
        # Preprocess training data
        X_train_processed = MODEL.named_steps['preprocessor'].transform(X_train)
        if hasattr(X_train_processed, 'toarray'):
            X_train_processed = X_train_processed.toarray()

        # Sample background data
        np.random.seed(42)
        sample_indices = np.random.choice(X_train_processed.shape[0], min(50, X_train_processed.shape[0]), replace=False)
        BACKGROUND_DATA = X_train_processed[sample_indices]
        
        app.logger.info(f"Background data shape: {BACKGROUND_DATA.shape}")
        
        # Create explainer
        EXPLAINER = shap.KernelExplainer(
            MODEL.named_steps['classifier'].predict_proba, 
            BACKGROUND_DATA
        )
        
        app.logger.info("SHAP Explainer created successfully")
        
    except Exception as e:
        app.logger.error(f"FATAL ERROR: {e}", exc_info=True)

with app.app_context():
    load_and_create_artifacts()

@app.route('/')
def index():
    """Renders the main HTML template."""
    if DATA_UI is None:
        return "Error: Backend data not loaded. Check model and DB connection.", 500

    select_options = {}
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
        select_options[col] = list(DATA_UI[col].dropna().unique())
        
    dependents_list = [str(x) for x in DATA_UI['Dependents'].dropna().unique()]
    select_options['Dependents'] = sorted(dependents_list, key=lambda x: int(x.replace('+', '')))
    
    return render_template('index.html', select_options=select_options)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Handles prediction and returns results including SHAP as Plotly chart."""
    global LAST_PREDICTION_DATA 
    
    if MODEL is None or EXPLAINER is None:
        return jsonify({'error': 'Model or Explainer not loaded. Check server startup logs.'}), 500

    try:
        form_data = request.json
        
        input_data = pd.DataFrame([{
            'Gender': form_data['gender'],
            'Married': form_data['married'],
            'Dependents': form_data['dependents'],
            'Education': form_data['education'],
            'Self_Employed': form_data['self_employed'],
            'ApplicantIncome': float(form_data.get('applicant_income', 0)),
            'CoapplicantIncome': float(form_data.get('coapplicant_income', 0.0)),
            'LoanAmount': float(form_data.get('loan_amount', 0.0)),
            'Loan_Amount_Term': float(form_data.get('loan_amount_term', 0.0)),
            'Credit_History': float(form_data.get('credit_history', 0.0)),
            'Property_Area': form_data['property_area']
        }])

        # Preprocess
        input_data_processed = MODEL.named_steps['preprocessor'].transform(input_data)
        if hasattr(input_data_processed, 'toarray'):
            input_data_processed = input_data_processed.toarray()
        
        app.logger.info(f"Input shape: {input_data_processed.shape}")
        
        # Predict
        prediction = MODEL.named_steps['classifier'].predict(input_data_processed)[0]
        prediction_proba = MODEL.named_steps['classifier'].predict_proba(input_data_processed)
        
        # Get feature names
        feature_names = MODEL.named_steps['preprocessor'].get_feature_names_out()
        
        # Calculate SHAP values
        app.logger.info("Calculating SHAP values...")
        shap_values_raw = EXPLAINER.shap_values(input_data_processed, nsamples=100)
        
        # Handle SHAP output
        class_arr = np.array(MODEL.named_steps['classifier'].classes_, dtype=object)
        matches = np.where(class_arr == prediction)[0]
        class_index = int(matches[0]) if matches.size > 0 else 0

        if isinstance(shap_values_raw, list):
            sample_shap = np.array(shap_values_raw[class_index]).flatten()
        else:
            sample_shap = np.array(shap_values_raw).flatten()
        
        app.logger.info(f"SHAP values shape: {sample_shap.shape}, Features: {len(feature_names)}")
        
        # Ensure lengths match
        min_length = min(len(sample_shap), len(feature_names), input_data_processed.shape[1])
        sample_shap = sample_shap[:min_length]
        usable_feature_names = feature_names[:min_length]
        features_series = pd.Series(input_data_processed.ravel()[:min_length], index=usable_feature_names)
        
        # Create SHAP DataFrame for visualization
        shap_df = pd.DataFrame({
            'Feature': usable_feature_names,
            'SHAP Value': sample_shap,
            'Feature Value': features_series.values
        })
        shap_df['Abs SHAP'] = np.abs(shap_df['SHAP Value'])
        shap_df = shap_df.sort_values('Abs SHAP', ascending=False).head(15)
        
        # Create Plotly bar chart for SHAP values
        colors = ['red' if x < 0 else 'blue' for x in shap_df['SHAP Value']]
        
        fig = go.Figure(data=[
            go.Bar(
                y=shap_df['Feature'],
                x=shap_df['SHAP Value'],
                orientation='h',
                marker=dict(color=colors),
                text=[f"Val: {v:.2f}" for v in shap_df['Feature Value']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<br>Feature Value: %{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Top 15 Feature Contributions to Prediction',
            xaxis_title='SHAP Value (Impact on Prediction)',
            yaxis_title='Feature Name',
            yaxis={'categoryorder':'total ascending'},
            height=500,
            margin=dict(l=200, r=50, t=50, b=50)
        )
        
        shap_html = fig.to_json()
        
        probability = float(prediction_proba[0][class_index])
        pred_status = "Approved" if (prediction == 1 or prediction == 'Y') else "Rejected"
        
        LAST_PREDICTION_DATA = {
            'status': pred_status,
            'probability': probability,
            'shap_values_dict': dict(zip(usable_feature_names, sample_shap)),
        }

        return jsonify({
            'status': 'success',
            'predicted_class': str(prediction),
            'predicted_status': pred_status,
            'probability': f"{probability*100:.2f}%",
            'shap_html': shap_html
        })
        
    except Exception as e:
        app.logger.error(f"Prediction Error: {e}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/save_prediction', methods=['POST'])
def api_save_prediction():
    """Saves the last prediction data to the database."""
    global LAST_PREDICTION_DATA
    if not LAST_PREDICTION_DATA:
        return jsonify({'status': 'error', 'message': 'No prediction data available to save.'}), 400

    success, message = save_prediction_to_db(
        application_id=1, 
        status=LAST_PREDICTION_DATA['status'], 
        prob=LAST_PREDICTION_DATA['probability'], 
        shap_values_dict=LAST_PREDICTION_DATA['shap_values_dict']
    )
    
    LAST_PREDICTION_DATA = {}
    return jsonify({'status': 'success' if success else 'error', 'message': message})

@app.route('/api/eda')
def api_eda():
    """Generates and returns Plotly charts for EDA as JSON."""
    if DATA_UI is None:
        return jsonify({'error': 'Data not loaded.'}), 500

    try:
        train_data_for_eda = DATA_UI.copy()
        train_data_for_eda = train_data_for_eda.dropna(subset=['Loan_Status'])
        train_data_for_eda['Loan_Status_Binary'] = train_data_for_eda['Loan_Status'].map({'Y': 1, 'N': 0})
        
        approval_rate = train_data_for_eda['Loan_Status_Binary'].mean() * 100
        avg_loan_amnt = train_data_for_eda['LoanAmount'].mean()
        avg_income = train_data_for_eda['ApplicantIncome'].mean()
        
        charts = {}
        for cat_feature in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
            fig = px.histogram(train_data_for_eda, x=cat_feature, color='Loan_Status', barmode='group',
                                title=f'Loan Status by {cat_feature}',
                                labels={'Loan_Status': 'Loan Status'})
            charts[cat_feature] = fig.to_json()

        return jsonify({
            'status': 'success',
            'metrics': {
                'approval_rate': f"{approval_rate:.2f}%",
                'avg_loan_amnt': f"${avg_loan_amnt:,.0f}",
                'avg_income': f"${avg_income:,.0f}",
            },
            'charts': charts,
        })
    except Exception as e:
        app.logger.error(f"EDA error: {e}")
        return jsonify({'error': f'EDA data generation failed: {e}'}), 500

@app.route('/api/history')
def api_history():
    """Fetches prediction history and generates charts."""
    engine = get_db_engine()
    if not engine:
        return jsonify({'error': 'Database connection failed.'}), 500

    try:
        query = """
        SELECT 
            p.Prediction_Id, p.Application_Id, p.Predicted_Status,
            p.Probability, p.Prediction_Date
        FROM Predictions p
        ORDER BY p.Prediction_Date DESC;
        """
        history_df = pd.read_sql(query, engine)
        
        if history_df.empty:
            return jsonify({'status': 'empty', 'message': 'No predictions found in the database yet.'})

        total_predictions = len(history_df)
        approval_rate = (history_df['Predicted_Status'].value_counts().get('Approved', 0) / total_predictions) * 100
        
        fig_pie = px.pie(history_df, names='Predicted_Status', title='Distribution of Approved vs. Rejected Predictions', 
                         color='Predicted_Status', color_discrete_map={'Approved':'#00B09B', 'Rejected':'#F5567B'})
        
        display_df = history_df.head(20).copy()
        display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x*100:.2f}%")
        display_df['Prediction_Date'] = pd.to_datetime(display_df['Prediction_Date']).dt.strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({
            'status': 'success',
            'metrics': {
                'total': total_predictions,
                'approval_rate': f"{approval_rate:.2f}%"
            },
            'pie_chart': fig_pie.to_json(),
            'recent_predictions': display_df.to_dict(orient='records')
        })
    except Exception as e:
        app.logger.error(f"History error: {e}")
        return jsonify({'error': f'Prediction history failed: {e}'}), 500

@app.route('/api/importance')
def api_importance():
    """Calculates and returns Global Feature Importance chart."""
    engine = get_db_engine()
    if not engine:
        return jsonify({'error': 'Database connection failed.'}), 500
        
    try:
        query = """
        SELECT 
            Feature_Name,
            AVG(ABS(Shap_Value)) AS mean_abs_shap_value
        FROM Explanations
        GROUP BY Feature_Name
        ORDER BY mean_abs_shap_value DESC;
        """
        feature_importance_df = pd.read_sql(query, engine)

        if feature_importance_df.empty:
            return jsonify({'status': 'empty', 'message': 'No explanation data found. Please make and save a few predictions first.'})

        fig = px.bar(
            feature_importance_df,
            x='mean_abs_shap_value',
            y='Feature_Name',
            orientation='h',
            title='Global Feature Importance',
            labels={'mean_abs_shap_value': 'Mean Absolute SHAP Value (Impact)', 'Feature_Name': 'Feature'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})

        return jsonify({
            'status': 'success',
            'bar_chart': fig.to_json()
        })
    except Exception as e:
        app.logger.error(f"Importance error: {e}")
        return jsonify({'error': f'Global importance failed: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
