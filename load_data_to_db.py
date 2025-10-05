import pandas as pd
from sqlalchemy import create_engine, text
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
        
        # Check if data already exists
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM customers"))
            count = result.scalar()
            
            if count > 0:
                print(f"\nWarning: Database already contains {count} customers.")
                print("Reloading data will:")
                print("  - Clear all existing customers and loan applications")
                print("  - Clear all predictions and SHAP explanations")
                response = input("\nDo you want to clear and reload data? (yes/no): ")
                
                if response.lower() != 'yes':
                    print("Migration cancelled. Existing data preserved.")
                    sys.exit(0)
                
                # Truncate all tables
                print("\nClearing tables...")
                with engine.begin() as conn_trans:
                    conn_trans.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
                    conn_trans.execute(text("TRUNCATE TABLE explanations"))
                    conn_trans.execute(text("TRUNCATE TABLE predictions"))
                    conn_trans.execute(text("TRUNCATE TABLE loan_applications"))
                    conn_trans.execute(text("TRUNCATE TABLE customers"))
                    conn_trans.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
                print("Tables cleared successfully.")
        
        print("Database connection successful.")
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        sys.exit(1)

    try:
        df = pd.read_csv(TRAIN_CSV_PATH)
        df.rename(columns={
            'Loan_ID': 'application_id_temp', 
            'Gender': 'Gender', 
            'Married': 'Married',
            'Dependents': 'Dependents', 
            'Education': 'Education', 
            'Self_Employed': 'Self_Employed',
            'ApplicantIncome': 'Income', 
            'LoanAmount': 'LoanAmount', 
            'Loan_Amount_Term': 'Loan_Amount_Term',
            'Credit_History': 'Credit_History', 
            'Property_Area': 'Property_Area', 
            'Loan_Status': 'Status'
        }, inplace=True)
        
        df['Dependents'] = df['Dependents'].replace('3+', '3')
        print("CSV data loaded and cleaned.")
        
    except FileNotFoundError:
        print(f"CSV file not found: {TRAIN_CSV_PATH}")
        print("Please ensure the file exists in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        sys.exit(1)

    # Create customers dataframe
    customers_df = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Income']].copy()
    customers_df.drop_duplicates(inplace=True)
    customers_df.reset_index(drop=True, inplace=True)
    customers_df['customer_id'] = customers_df.index + 1

    # Merge to get customer_id for each loan application
    df = pd.merge(
        df, 
        customers_df, 
        on=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Income'], 
        how='left'
    )
    
    loan_applications_df = df[['customer_id', 'LoanAmount', 'Loan_Amount_Term', 'Property_Area', 'Credit_History', 'Status']].copy()
    print("Dataframes for database tables created.")
    
    try:
        print("\nInserting data into 'customers' table...")
        customers_df.to_sql('customers', con=engine, if_exists='append', index=False)
        print(f"Successfully inserted {len(customers_df)} customers.")

        print("Inserting data into 'loan_applications' table...")
        loan_applications_df.to_sql('loan_applications', con=engine, if_exists='append', index=False)
        print(f"Successfully inserted {len(loan_applications_df)} loan applications.")

    except Exception as e:
        print(f"\nAn error occurred during database insertion: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure your MySQL tables have the correct schema")
        print("2. Check that foreign key constraints are properly set up")
        print("3. Verify database user permissions")
        sys.exit(1)

    print("\n--- Data Migration Complete! ---")
    print(f"Total customers: {len(customers_df)}")
    print(f"Total loan applications: {len(loan_applications_df)}")

if __name__ == "__main__":
    migrate_csv_to_db()
