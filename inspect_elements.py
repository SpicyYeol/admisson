import pandas as pd
import sys

# Set encoding to utf-8 for output
sys.stdout.reconfigure(encoding='utf-8')

try:
    file_path = r'c:\Users\wagon\Documents\admission\ref.xlsx'
    df = pd.read_excel(file_path, sheet_name='전형요소')
    print("Columns:")
    for col in df.columns:
        print(f"- {col}")
        
    print("\nSample Data (Filter '면접'):")
    # Clean column names just in case
    df.columns = [c.strip() for c in df.columns]
    
    # Try to find the admission name column
    name_col = [c for c in df.columns if '전형명' in c][0]
    
    # Load Student Data (Sheet 1)
    df_std = pd.read_excel(file_path, engine='openpyxl') # Default sheet
    std_types = set(df_std['전형구분'].astype(str).unique())
    
    # Load Elements Data
    df_ele = pd.read_excel(file_path, sheet_name='전형요소')
    # Clean column names
    df_ele.columns = [c.strip() for c in df_ele.columns]
    # Find name col
    name_col = [c for c in df_ele.columns if '전형명' in c][0]
    ele_types = set(df_ele[name_col].astype(str).unique())
    
    print(f"Student Data Types: {len(std_types)}")
    print(f"Element Data Types: {len(ele_types)}")
    print(f"Intersection: {len(std_types & ele_types)}")
    
    missing_in_ele = std_types - ele_types
    print(f"\nMissing in Elements (Top 5): {list(missing_in_ele)[:5]}")
except Exception as e:
    print(e)
