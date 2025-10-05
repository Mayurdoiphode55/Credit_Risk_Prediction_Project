# 🏦 Credit Risk Prediction System with MLOps & Explainable AI (Flask/MySQL)

This project is an end-to-end machine learning application designed to predict credit risk for loan applications. It integrates a powerful ML model with a full database backend and an interactive **RESTful API and web dashboard built with Flask**, demonstrating a full MLOps cycle from data loading to prediction, explanation, and monitoring.

***
## ✨ Key Features

* **RESTful API & Web Dashboard**: A lightweight, high-performance web application built with **Flask** serving both a dynamic HTML dashboard (using Jinja templates) and a set of **REST APIs** for predictions and data retrieval.
* **High-Accuracy ML Model**: Utilizes an **XGBoost** classifier, trained on historical loan data, for robust and reliable predictions.
* **Full Database Integration**: A **MySQL** database backend stores all customer data, loan applications, predictions, and model explanations, making the system persistent and scalable.
* **Explainable AI (XAI)**:
    * **Local Explanations**: Generates **SHAP force plots** (rendered via Plotly/JSON) for every prediction, showing exactly which features contributed to the loan approval or rejection decision.
    * **Global Explanations**: A dashboard page queries historical SHAP values from the database, revealing the most influential features for the model's behavior over time.
* **Prediction Monitoring**: A "Prediction History" dashboard page queries the database to display key metrics, prediction distributions, and a log of the most recent decisions made by the model.
* **Automated Data Loading**: A Python script is provided to automatically migrate the initial dataset from a CSV file into the normalized relational database schema.

***
## 🛠️ Tech Stack

| Category | Tools Used |
| :--- | :--- |
| **Web & API** | **Flask**, Jinja2, RESTful APIs |
| **Backend & ML** | Python, Pandas, NumPy, Scikit-learn |
| **ML Model** | XGBoost |
| **Explainability** | SHAP |
| **Database** | MySQL |
| **DB Connector** | SQLAlchemy, PyMySQL |
| **Plotting** | Plotly (for dynamic charts) |

***
## 📈 Model Performance

The XGBoost model was evaluated on the test set to assess its performance in predicting loan approval status. The model demonstrates strong predictive power with an overall accuracy of **84%**.

**Classification Report:**

| | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Rejected)** | 0.95 | 0.55 | 0.70 | 38 |
| **1 (Approved)** | 0.80 | 0.98 | 0.88 | 85 |
| **Accuracy** | | | **0.84** | 123 |

### Key Observations:
* **High Accuracy**: The model achieves an **overall accuracy of 84%**.
* **Excellent Recall for Approvals**: A **recall of 0.98** for the "Approved" class (1) means the model rarely misses a qualified applicant, minimizing opportunity cost.
* **Strong Precision for Rejections**: A **precision of 0.95** for the "Rejected" class minimizes the risk of rejecting creditworthy applicants incorrectly.

***
## 🏗️ System Architecture, Schema, and Setup

This section covers the system's architecture, database design, project layout, and instructions for getting it running.

### System Architecture

The application uses a standard three-tier architecture where the Flask application handles both the presentation layer (HTML/Jinja) and the business logic (ML/API), interfacing directly with the database.

```mermaid
graph TD
    A[User] -->|Interacts with| B(Flask Web Application);
    B -->|Serves HTML Templates| A;
    B -->|Calls API Endpoints| B;
    B -->|Loads Model at Startup| D([Models/risk_model.joblib]);
    B -->|Reads/Writes Data| E[(MySQL Database)];
    B -->|Generates SHAP Explanation| B;

### 🗂️ Database Schema (ERD)
The database is designed with four normalized tables to efficiently store all relevant information.
   erDiagram
    customers {
        INT customer_id PK
        VARCHAR Gender
        VARCHAR Married
        VARCHAR Dependents
        VARCHAR Education
        VARCHAR Self_Employed
        FLOAT Income
    }
    loan_applications {
        INT application_id PK
        INT customer_id FK
        FLOAT LoanAmount
        INT Loan_Amount_Term
        VARCHAR Property_Area
        BOOLEAN Credit_History
        VARCHAR Status
    }
    predictions {
        INT Prediction_Id PK
        INT Application_Id FK
        VARCHAR Predicted_Status
        FLOAT Probability
        TIMESTAMP Prediction_Date
    }
    explanations {
        INT Explanation_Id PK
        INT Prediction_Id FK
        VARCHAR Feature_Name
        FLOAT Shap_Value
    }
    customers ||--|{ loan_applications : "submits"
    loan_applications ||--o{ predictions : "receives"
    predictions ||--|{ explanations : "is composed of"

---
### 🗂️ Project Structure
The repository is organized for clarity and MLOps best practices.
   
