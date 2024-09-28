
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def impute_group_mean(dataframe, group_cols, target_col):
    """
    Impute missing values in the target column using the mean of the group defined by group_cols.
    
    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    group_cols (list): The columns used to define the groups.
    target_col (str): The target column where missing values will be imputed.
    
    Returns:
    pd.DataFrame: The DataFrame with imputed values.
    """
    dataframe[target_col] = dataframe.groupby(group_cols)[target_col].transform(
        lambda x: x.fillna(x.mean()))
    return dataframe

def apply_imputations(dataframe):
    """
    Apply imputations to the dataframe using group means and overall means.
    
    Parameters:
    dataframe (pd.DataFrame): The DataFrame to be imputed.
    
    Returns:
    pd.DataFrame: The DataFrame after imputation.
    """
    num_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns
   
    for col in num_cols:
        dataframe = impute_group_mean(dataframe, ['Local authority code'], col)
        dataframe = impute_group_mean(dataframe, ['Region code'], col)

    imputer = SimpleImputer(strategy='mean')
    dataframe[num_cols] = imputer.fit_transform(dataframe[num_cols])

    return dataframe

def prepare_data(dataframe, target_column='Price'):
    """
    Prepare the data by applying final imputations and splitting into train and test sets.
    
    Parameters:
    dataframe (pd.DataFrame): The DataFrame to be processed.
    target_column (str): The name of the target variable column.
    
    Returns:
    tuple: Contains the transformed DataFrame, training and testing datasets (X_train, X_test, y_train, y_test).
    """
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    imputer = SimpleImputer(strategy='mean')
    
    # Imputing missing values in features and convert back to DataFrame to maintain column names
    X_imputed = imputer.fit_transform(X)
    dataframe = pd.DataFrame(X_imputed, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return dataframe, X, y, X_train, X_test, y_train, y_test

