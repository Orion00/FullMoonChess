# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from datetime import datetime, time

# %%
crime = pd.read_csv("crimes_cleaned.csv")
crime['Arrest'] = crime['Arrest'].astype(int)
crime['Domestic'] = crime['Domestic'].astype(int)
crime['Ward'] = crime['Ward'].astype('category')
crime['Community Area'] = crime['Community Area'].astype('category')
crime['Beat'] = crime['Beat'].astype('category')
crime['new_date'] = pd.to_datetime(crime['new_date'])
crime['Time'] = pd.to_datetime(crime['Time'], format='%H:%M:%S').dt.time
crime = crime.drop('Date',axis=1)

# %%
crime.dtypes

# %%
X_train, X_test, y_train, y_test = train_test_split(crime.drop('Primary Type',axis=1), crime['Primary Type'],random_state=101, train_size=.8)

# %%
numeric_columns = selector(dtype_include=['number'])(X_train)
categorical_columns = selector(dtype_include=['object'])(X_train)

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('scalar', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(max_depth=150, random_state=101, n_jobs=-1))
])

# %%
crime['Primary Type'].value_counts() < 5