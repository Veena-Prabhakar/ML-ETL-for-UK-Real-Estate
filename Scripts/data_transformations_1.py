
import pandas as pd

def print_header(header):
    """Prints a bold header for each section."""
    print(f"\n\033[1m{header}\033[0m")

def drop_unnecessary_columns(df):
    """Drops column that are no longer needed."""
    df = df.drop(columns=['extraction_timestamp'])
    print_header("Dropping Unnecessary column:")
    print("Dropped 'extraction_timestamp' column.")
    return df

def check_duplicates(df):
    """Checks for duplicate rows in the DataFrame."""
    print_header("Duplicate check:")
    duplicate_rows = df.duplicated().any()
    if duplicate_rows:
        print("There are duplicate rows in the DataFrame.")
    else:
        print("No duplicate rows found in the DataFrame.")

def rename_columns(df):
    """Renames column according to the specified mapping."""
    print_header("Column Renaming:")
    df.rename(columns={
        'HGV - \nMotorways': 'HGV - Motorways',
        'Freight transport (HGV and LGV)\n[Note 5]': 'Freight transport (HGV and LGV)',
        'Headcount of Pupils': 'Headcount of Pupils(school)',
        'No qualifications (number)': 'No qualifications',
        'Level 1 and entry level qualifications (number)': 'Level 1 and entry level qualifications',
        'Level 2 qualifications (number)': 'Level 2 qualifications',
        'Apprenticeship (number)': 'Apprenticeship',
        'Level 3 qualifications (number)': 'Level 3 qualifications',
        'Level 4 qualifications and above (number)': 'Level 4 qualifications and above',
        'Other (number)': 'Other qualifications',
        'Qualification index rank \n(lowest = 1, highest = 331)': 'Qualification index rank (1 to 331)',
        'Region code_x': 'Region code',
        'Region name_x': 'Region name',
        'Local authority code ': 'Local authority code',
        'One Bedroom': 'One Bedroom Rent',
        'Two Bedrooms': 'Two Bedrooms Rent',
        'Three Bedrooms': 'Three Bedrooms Rent',
        'Four or more Bedrooms': 'Four or more Bedrooms Rent',
        'All categories': 'All categories Rent'
    }, inplace=True)
    print("Renamed columns.")
    return df

def check_null_records(df):
    """Checks for null record and print the number of null records per column."""
    print_header("Null values in Each record:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        print(f"Column: {col}, Number of Null Records: {count}")
    total_null_records = null_counts.sum()
    print(f"Total Number of Null Records in DataFrame: {total_null_records}")

def clean_data(df):
    """Removes record with empty values for key attributes and display record count."""
    print_header("Drop records with Null values on Key attributes:")
    initial_count = len(df)
    df = df.dropna(subset=['AveragePrice', 'Local authority code', 'Town/City'])
    cleaned_count = len(df)
    print(f"Number of records before cleaning: {initial_count}")
    print(f"Number of records after cleaning: {cleaned_count}")
    return df

def save_to_excel(df, filepath):
    """Saves the cleaned DataFrame to an Excel file."""
    print_header("Data export:")
    df.to_excel(filepath, index=False)
    print(f"Data saved to '{filepath}'")

def main_processing_pipeline(file_path):
    """Main processing pipeline that runs all steps in sequence."""
    df = pd.read_excel(file_path)
    df = drop_unnecessary_columns(df)
    check_duplicates(df)
    df = rename_columns(df)
    check_null_records(df)
    df = clean_data(df)
    save_to_excel(df, "../Data/Output/original_cleaned_df.xlsx")
    return df
