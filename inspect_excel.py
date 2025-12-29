import sys
import os

print("Python Executable:", sys.executable)
# print("Sys Path:", sys.path)

try:
    import openpyxl
    print("openpyxl imported successfully")
except ImportError as e:
    print(f"Failed to import openpyxl: {e}")

try:
    import pandas as pd
    print("pandas imported successfully")
    
    file_path = r'c:\Users\wagon\Documents\admission\ref.xlsx'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            print("Columns:")
            print(df.columns.tolist())
            print("\nFirst 5 rows:")
            print(df.head())
            print("\nData Types:")
            print(df.dtypes)
        except Exception as e:
            print(f"Error reading excel file: {e}")

except ImportError as e:
    print(f"Failed to import pandas: {e}")
