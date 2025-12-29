import pandas as pd

file_path = r'c:\Users\wagon\Documents\admission\ref.xlsx'
try:
    df = pd.read_excel(file_path, engine='openpyxl')
    print("Unique 합격구분:", df['합격구분'].unique())
    print("\nUnique 등록구분:", df['등록구분'].unique())
    
    # Also check if 0 exists in grades
    print("\nZero counts in 수능등급:", (df['수능등급'] == 0).sum())
    print("Zero counts in 석차백분율(내신):", (df['석차백분율(내신)'] == 0).sum())
except Exception as e:
    print(e)
