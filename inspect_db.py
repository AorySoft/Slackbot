import sqlite3
import os

def inspect_database():
    """Inspect the database and print all tables and their contents"""
    db_path = "slack_bot.db"
    
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist!")
        return
    
    print(f"\n=== Database Inspection ===")
    print(f"Database file: {db_path}")
    print(f"File size: {os.path.getsize(db_path)} bytes")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\nFound {len(tables)} tables:")
        for table in tables:
            table_name = table[0]
            print(f"- {table_name}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            print(f"  Row count: {row_count}")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print(f"  Columns: {len(columns)}")
            for col in columns:
                print(f"    - {col[1]} ({col[2]})")
            
            # Get sample data (first 5 rows)
            if row_count > 0:
                print(f"  Sample data (up to 5 rows):")
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                rows = cursor.fetchall()
                for row in rows:
                    print(f"    {row}")
            
            print()
        
        conn.close()
    except Exception as e:
        print(f"Error inspecting database: {str(e)}")

if __name__ == "__main__":
    inspect_database() 