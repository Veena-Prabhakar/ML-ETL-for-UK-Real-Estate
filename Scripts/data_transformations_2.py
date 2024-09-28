
import pandas as pd
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

def drop_columns1(X):
    """
    Drops specified column from the DataFrame.
    
    Parameters:
    X (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with specified columns dropped.
    """
    return X.drop(columns=['Number of Schools', 'Headcount of Pupils(school)', 'NewSalesVolume'])

class GroupMeanImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that imputes missing values in specified target columns
    by calculating the mean within groups defined by specified columns.
    """
    def __init__(self, group_cols, target_cols):
        self.group_cols = group_cols
        self.target_cols = target_cols

    def fit(self, X, y=None):
        """
        Compute the mean values for the groups in the target columns during fitting.
        
        """
        self.group_means_ = X.groupby(self.group_cols)[self.target_cols].transform('mean')
        return self

    def transform(self, X):
        """
        Impute missing values in the target columns by replacing NaNs with the mean
        values computed during the fitting process.
        """
        for col in self.target_cols:
            if X[col].isnull().any():  # Proceed only if there are missing values
                X[col] = X[col].fillna(X.groupby(self.group_cols)[col].transform('mean'))
        return X

class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Custom transformer for creating new features from existing columns in a DataFrame.
    This includes extracting time-based features and calculating specific ratios and percentage changes.
    """
    def fit(self, X, y=None):
        """
        Fits the transformer, ensuring that the input is a pandas DataFrame and capturing the column names.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame.
        y: Ignored, present for compatibility.
        
        Returns:
        self: Returns the instance itself.
        """
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.tolist()
        else:
            raise ValueError("Input must be a pandas DataFrame during fitting.")
        return self

    def transform(self, X):
        """
        Transforms the input DataFrame by adding new features such as date-based components, 
        price ratios, and percentage changes, as well as applying logarithmic transformations.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame.
        
        Returns:
        X (pd.DataFrame): The transformed DataFrame with new features.
        """
        
        X['Date'] = pd.to_datetime(X['Date'])
        
        # Extracting month, quarter, and year from 'Date'
        X['Month'] = X['Date'].dt.month
        X['Quarter'] = X['Date'].dt.quarter
        X['Year'] = X['Date'].dt.year

        # Creating specific price ratios as per manual steps
        X['Detached_SemiDetached_Ratio'] = X['DetachedPrice'] / X['SemiDetachedPrice']
        X['Detached_Terraced_Ratio'] = X['DetachedPrice'] / X['TerracedPrice']
        X['Detached_Flat_Ratio'] = X['DetachedPrice'] / X['FlatPrice']
        
        # Defining specific price columns for percentage changes and ratios
        price_columns = ['AveragePrice', 'DetachedPrice', 'SemiDetachedPrice', 'TerracedPrice', 'FlatPrice']
        
        # Calculating percentage changes for price-related columns
        for col in price_columns:
            X[col + '_PctChange'] = X[col].pct_change()
        
        # Applying log transformation to the same price columns
        log_transform_cols = ['SalesVolume', 'DetachedPrice', 'SemiDetachedPrice', 'TerracedPrice', 'FlatPrice', 'AveragePrice']
        for col in log_transform_cols:
            X[col + '_log'] = np.log1p(X[col])
        
        return X

def cap_floor(df, lower_quantile=0.05, upper_quantile=0.95):
    """
    Apply capping and flooring to numerical features in the DataFrame based on specified quantiles.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to apply capping and flooring.
    lower_quantile (float): The quantile value below which data will be floored.
    upper_quantile (float): The quantile value above which data will be capped.
    
    Returns:
    pd.DataFrame: The DataFrame with capped and floored values.
    """
    df_capped = df.copy() 
    for col in df_capped.select_dtypes(include=[np.number]).columns:
        lower_bound = df_capped[col].quantile(lower_quantile)
        upper_bound = df_capped[col].quantile(upper_quantile)
        
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_capped

def cap_floor_func(X):
    """
    Wrapper function for applying capping and flooring within a pipeline.
    
    Parameters:
    X (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with capped and floored values.
    """
    return cap_floor(X, lower_quantile=0.05, upper_quantile=0.95)

class EncodingWithNames(BaseEstimator, TransformerMixin):
    """
    A custom transformer that performs ordinal encoding on specified columns while
    preserving column names and handling passthrough columns.
    """
    def __init__(self, columns, remainder='passthrough'):
        self.columns = columns
        self.remainder = remainder
        self.encoder = ColumnTransformer(
            transformers=[
                ('ordinal_encoder', OrdinalEncoder(), columns)
            ],
            remainder=remainder,
            verbose_feature_names_out=False  # Prevent auto-renaming
        )

    def fit(self, X, y=None):
        """
        Fit the internal ColumnTransformer on the input data.
        """
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        """
        Transform the data, encoding specified columns and combining them with passthrough data.
        """
        transformed = self.encoder.transform(X)
        if isinstance(transformed, np.ndarray):
            if self.remainder == 'passthrough':
                all_columns = self.columns + [col for col in X.columns if col not in self.columns]
            else:
                all_columns = self.columns
            return pd.DataFrame(transformed, index=X.index, columns=all_columns)
        return transformed

class ColumnTransformerDf(ColumnTransformer, TransformerMixin):
    """
    A custom ColumnTransformer that ensures the output is a DataFrame with appropriate
    column names and index, maintaining the structure of the original DataFrame.
    """
    def transform(self, X):
        """
        Transform the data and ensure that the result is returned as a DataFrame.
        """
        result = super().transform(X)
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result, index=X.index, columns=self.get_feature_names_out())
        return result

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it, ensuring the output is a DataFrame.
        """
        result = super().fit_transform(X, y=y)
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result, index=X.index, columns=self.get_feature_names_out())
        return result

class ColumnOrderTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that reorders columns in a DataFrame according to original order to maintain consistency after different pipeline process.
    """
    def __init__(self, column_order):
        self.column_order = column_order

    def fit(self, X, y=None):
        """
        No fitting necessary; return self.
        """
        return self

    def transform(self, X):
        """
        Reorder the DataFrame columns based on the specified column order.
        """
        return X[self.column_order]

# Custom function to drop specified columns from the DataFrame
def drop_columns2(X):
    """
    Drops column that are unnecessary for the analysis and model training.
    
    Parameters:
    X (pd.DataFrame): The input DataFrame.
    
    Returns:
    X (pd.DataFrame): The DataFrame with specified columns removed.
    """
    return X.drop(columns=['District', 'Transfer Month-Year', 'Town/City', 'County', 'Old/New',
                           'Record Status', 'Region name', 'Local authority name', 'RegionName', 'AreaCode', 'Date'])

# Custom function for manipulating specific feature values for removing prefixes from strings
def feature_manipulation(X):
    """
    Performs specific feature manipulation to remove the prefix 'E' from the region and local authority codes and convert data types.
    
    Parameters:
    X (pd.DataFrame): The input DataFrame.
    
    Returns:
    X (pd.DataFrame): The DataFrame with manipulated feature values.
    """
    X = X.copy()
    for col in ['Region code', 'Local authority code']:
        X[col] = X[col].str.replace('E', '').astype(str)
    return X

# Custom function to handle infinite values by replacing them with NaNs
def handle_infinite_values(X):
    """
    Replaces infinite value in the DataFrame with NaNs to handle them appropriately in subsequent steps.
    
    Parameters:
    X (pd.DataFrame): The input DataFrame.
    
    Returns:
    X (pd.DataFrame): The DataFrame with infinite values replaced by NaNs.
    """
    return X.replace([np.inf, -np.inf], np.nan)


# Defining multiple group mean imputers for different sets of group-by columns and target columns

# GroupMeanImputer for imputing missing values based on region and local authority
group_mean_imputer1 = GroupMeanImputer(
    group_cols=['Region code', 'Region name', 'Local authority code', 'Local authority name'],
    target_cols=['SalesVolume', 'DetachedPrice', 'SemiDetachedPrice', 'TerracedPrice', 'NewPrice', 'OldPrice', 
                 'OldSalesVolume', 'Annual change (%)', 'Rental price (£)', 'One Bedroom Rent', 'Two Bedrooms Rent', 
                 'Three Bedrooms Rent', 'Four or more Bedrooms Rent', 'All categories Rent', 
                 'Qualification index score', 'Qualification index rank (1 to 331)', 'No qualifications', 
                 'Level 1 and entry level qualifications', 'Level 2 qualifications', 'Apprenticeship', 
                 'Level 3 qualifications', 'Level 4 qualifications and above', 'Other qualifications', 
                 'Estimated number of households with at least 1 early-years or school age child', 
                 'Deprivation Average Score', 'Number of those aged 16+ who are unemployed', 
                 'Number of those aged 16+ in employment who are employees',
                 'Number of those aged 16+ in employment who are self-employed','GDHI'])

# GroupMeanImputer for imputing missing values based on broader region grouping
group_mean_imputer2 = GroupMeanImputer(
    group_cols=['Region code', 'Region name'],
    target_cols=['DetachedPrice', 'SemiDetachedPrice', 'TerracedPrice', 'NewPrice', 
                'Annual change (%)', 'Rental price (£)', 'One Bedroom Rent', 'Two Bedrooms Rent', 
                'Three Bedrooms Rent', 'Four or more Bedrooms Rent', 'All categories Rent', 
                'Qualification index score', 'Qualification index rank (1 to 331)', 'No qualifications', 
                'Level 1 and entry level qualifications', 'Level 2 qualifications', 'Apprenticeship', 
                'Level 3 qualifications', 'Level 4 qualifications and above', 'Other qualifications', 
                'Estimated number of households with at least 1 early-years or school age child', 
                'Deprivation Average Score', 'Number of those aged 16+ who are unemployed', 
                'Number of those aged 16+ in employment who are employees', 
                'Number of those aged 16+ in employment who are self-employed', 'GDHI' ])

# Instantiating the custom FeatureCreator transformer    
feature_creation = FeatureCreator()

# Creating a FunctionTransformer that wraps the cap_floor function for capping and flooring values
cap_floor_transformer = FunctionTransformer(func=cap_floor_func, validate=False)

# Initializing the EncodingWithNames transformer to handle categorical encoding within the pipeline
encoding_transformer = EncodingWithNames(
    columns=['Property Type', 'Duration', 'PPD Category Type'],
    remainder='passthrough'
)

# Defining scalers for different groups of features
robust_scaler_price = RobustScaler()
standard_scaler = StandardScaler()

# Combining different transformations into a single ColumnTransformer for scaling and normalizing numerical features
# This applies RobustScaler to price-related columns (handling outliers) and StandardScaler to other columns for standard normalization.

scaling_normalizing_transformer = ColumnTransformerDf(transformers=[
    ('robust_scaler_price', RobustScaler(), ['Price']),
    ('robust_scaler_others', RobustScaler(), [    'SalesVolume', 'DetachedPrice', 'SemiDetachedPrice', 'TerracedPrice', 
                                                'FlatPrice', 'OldSalesVolume', 'Annual change (%)', 'All ages', 'Area (sq km)', 
                                                'Number of those aged 16+ in employment who are self-employed', 'Buses total', 
                                                'Diesel cars total', 'Petrol cars total', 'Freight transport (HGV and LGV)', 
                                                'Fuel consumption by all vehicles']),
    ('standard_scaler', StandardScaler(), [
        'AveragePrice', 'Index', '1m%Change', 'CashPrice', 'MortgagePrice', 'MortgageIndex', 
    'FTBPrice', 'FOOPrice', 'NewPrice', 'OldPrice', 'Rental price (£)', 'One Bedroom Rent', 
    'Two Bedrooms Rent', 'Three Bedrooms Rent', 'Four or more Bedrooms Rent', 'All categories Rent', 
    '0-20', '20-40', '40-60', '60+', 'Female population', 'Male population',
    'Qualification index rank (1 to 331)',
    'Qualification index score', 'No qualifications', 'Level 1 and entry level qualifications', 
    'Level 2 qualifications', 'Apprenticeship', 'Level 3 qualifications', 'Level 4 qualifications and above', 
    'Other qualifications', 'Estimated number of households with at least 1 early-years or school age child', 
    'Deprivation Average Score', 'Number of those aged 16+ who are unemployed', 
    'Number of those aged 16+ in employment who are employees', 'GDHI', 'Diesel LGV total', 
    'Petrol LGV total', 'LPG LGV total', 'Personal transport (buses, cars and motorcycles)'
    ]),
    ], remainder='passthrough', verbose_feature_names_out=False)

# List of columns in the desired order
desired_column_order = ['District', 'Transfer Month-Year', 'Town/City', 'County', 'Price',
       'Property Type', 'Old/New', 'Duration', 'PPD Category Type',
       'Record Status', 'Region code', 'Region name', 'Local authority code',
       'Local authority name', 'Date', 'RegionName', 'AreaCode',
       'AveragePrice', 'Index', '1m%Change', 'SalesVolume', 'DetachedPrice',
       'SemiDetachedPrice', 'TerracedPrice', 'FlatPrice', 'CashPrice',
       'MortgagePrice', 'MortgageIndex', 'FTBPrice', 'FOOPrice', 'NewPrice',
       'OldPrice', 'OldSalesVolume', 'Annual change (%)', 'Rental price (£)',
       'One Bedroom Rent', 'Two Bedrooms Rent', 'Three Bedrooms Rent',
       'Four or more Bedrooms Rent', 'All categories Rent', 'All ages', '0-20',
       '20-40', '40-60', '60+', 'Female population', 'Male population',
       'Area (sq km)', 'Qualification index score',
       'Qualification index rank (1 to 331)', 'No qualifications',
       'Level 1 and entry level qualifications', 'Level 2 qualifications',
       'Apprenticeship', 'Level 3 qualifications',
       'Level 4 qualifications and above', 'Other qualifications',
       'Estimated number of households with at least 1 early-years or school age child',
       'Deprivation Average Score',
       'Number of those aged 16+ who are unemployed',
       'Number of those aged 16+ in employment who are employees',
       'Number of those aged 16+ in employment who are self-employed', 'GDHI',
       'Buses total', 'Diesel cars total', 'Petrol cars total',
       'HGV - Motorways', 'HGV total', 'Diesel LGV total', 'Petrol LGV total',
       'LPG LGV total', 'Personal transport (buses, cars and motorcycles)',
       'Freight transport (HGV and LGV)', 'Fuel consumption by all vehicles',
       'Month', 'Quarter', 'Year', 'Detached_SemiDetached_Ratio',
       'Detached_Terraced_Ratio', 'Detached_Flat_Ratio',
       'AveragePrice_PctChange', 'DetachedPrice_PctChange',
       'SemiDetachedPrice_PctChange', 'TerracedPrice_PctChange',
       'FlatPrice_PctChange', 'SalesVolume_log', 'DetachedPrice_log',
       'SemiDetachedPrice_log', 'TerracedPrice_log', 'FlatPrice_log',
       'AveragePrice_log']
