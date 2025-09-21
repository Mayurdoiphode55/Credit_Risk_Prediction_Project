import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import sys

# --- DATABASE CONNECTION DETAILS ---
DB_CONFIG = {
    "drivername": "mysql+pymysql",
    "username": "root",
    "password": "Mayur9730@",
    "host": "127.0.0.1",
    "database": "credit_risk_db"
}

# --- PATH TO YOUR CSV FILE ---
TRAIN_CSV_PATH = 'train.csv'

def migrate_csv_to_db():
    print("--- Starting Data Migration ---")

    try:
        connection_url = URL.create(**DB_CONFIG)
        engine = create_engine(connection_url)
        print("✅ Database connection successful.")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

    try:
        df = pd.read_csv(TRAIN_CSV_PATH)
        df.rename(columns={
            'Loan_ID': 'application_id_temp', 'Gender': 'Gender', 'Married': 'Married',
            'Dependents': 'Dependents', 'Education': 'Education', 'Self_Employed': 'Self_Employed',
            'ApplicantIncome': 'Income', 'LoanAmount': 'LoanAmount', 'Loan_Amount_Term': 'Loan_Amount_Term',
            'Credit_History': 'Credit_History', 'Property_Area': 'Property_Area', 'Loan_Status': 'Status'
        }, inplace=True)
        df['Dependents'] = df['Dependents'].replace('3+', '3')
        print("✅ CSV data loaded and cleaned.")
    except Exception as e:
        print(f"❌ An error occurred while reading the CSV: {e}")
        sys.exit(1)

    customers_df = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Income']].copy()
    customers_df.drop_duplicates(inplace=True)
    customers_df.reset_index(drop=True, inplace=True)
    customers_df['customer_id'] = customers_df.index + 1

    df = pd.merge(df, customers_df, on=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Income'], how='left')
    loan_applications_df = df[['customer_id', 'LoanAmount', 'Loan_Amount_Term', 'Property_Area', 'Credit_History', 'Status']].copy()
    print("✅ Dataframes for database tables created.")
    
    try:
        # We now use 'append' because the tables already exist
        print("Inserting data into 'customers' table...")
        customers_df.to_sql('customers', con=engine, if_exists='append', index=False)
        print("✅ Data successfully inserted into 'customers'.")

        print("Inserting data into 'loan_applications' table...")
        loan_applications_df.to_sql('loan_applications', con=engine, if_exists='append', index=False)
        print("✅ Data successfully inserted into 'loan_applications'.")

    except Exception as e:
        # If you ever see a "Duplicate entry" error here, it means you forgot
        # to DROP the tables in MySQL before running the script.
        print(f"❌ An error occurred during database insertion: {e}")
        sys.exit(1)

    print("\n🎉 --- Data Migration Complete! --- 🎉")

if __name__ == "__main__":
    migrate_csv_to_db()