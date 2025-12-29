import pandas as pd

file_path = r'c:\Users\wagon\Documents\admission\ref.xlsx'
try:
    df = pd.read_excel(file_path, engine='openpyxl')
    print("Columns:", df.columns.tolist())
    print("\nUnique 합격구분:", df['합격구분'].unique())
    print("\nUnique 전형구분 (Sample):", df['전형구분'].unique())
    
    # Check if we can deduce factors. Maybe only by name?
    # No explicit columns like 'Exam%', 'Intreview%' visible in previous turns.
    
except Exception as e:
    print(e)
