
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def check_db():
    try:
        db = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "fraud_db")
        )
        print("Connected to database successfully.")
        cur = db.cursor()
        cur.execute("SHOW TABLES LIKE 'transactions'")
        table = cur.fetchone()
        if table:
            print("Table 'transactions' exists.")
        else:
            print("Table 'transactions' does NOT exist.")
            # Attempt to create table
            create_table_sql = """
            CREATE TABLE transactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255),
                amount FLOAT,
                latitude FLOAT,
                longitude FLOAT,
                city_pop FLOAT,
                unix_time FLOAT,
                merch_lat FLOAT,
                merch_long FLOAT,
                risk_score FLOAT,
                status VARCHAR(50),
                is_fraud TINYINT(1)
            )
            """
            cur.execute(create_table_sql)
            print("Table 'transactions' created.")
        cur.close()
        db.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db()
