import pandas as pd

file_path = r'c:\Users\wagon\Documents\admission\ref.xlsx'
try:
    df = pd.read_excel(file_path, engine='openpyxl')
    print("Columns:", df.columns.tolist())
    
    # Check for '면접' in column names
    interview_cols = [c for c in df.columns if '면접' in c]
    print("Interview Columns:", interview_cols)
    
    # Check unique '전형구분' to guess
    print("Unique 전형구분:", df['전형구분'].unique())
    
except Exception as e:
    print(e)
