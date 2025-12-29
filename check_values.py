import pandas as pd

file_path = r'c:\Users\wagon\Documents\admission\ref.xlsx'
try:
    df = pd.read_excel(file_path, engine='openpyxl')
    print("Unique 모집구분:", df['모집구분'].unique())
    print("\nUnique 전형구분:", df['전형구분'].unique())
    print("\nUnique 교육청소재지 (First 10):", df['교육청소재지'].unique()[:10])
except Exception as e:
    print(e)
