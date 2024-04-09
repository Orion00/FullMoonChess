# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from datetime import datetime, time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import shap
import numpy as np
import matplotlib.pyplot as plt

# %%
crime = pd.read_csv("aggregated_crimes.csv")
#crime['Arrest'] = crime['Arrest'].astype(int)
#crime['Domestic'] = crime['Domestic'].astype(int)
crime['Ward'] = crime['Ward'].astype('category')
#crime['Community Area'] = crime['Community Area'].astype('category')
#crime['Beat'] = crime['Beat'].astype('category')
#crime['new_date'] = pd.to_datetime(crime['new_date'])
#crime['Time'] = pd.to_datetime(crime['Time'], format='%H:%M:%S').dt.time
crime = crime.drop('new_date',axis=1)
columns_to_convert = ['Holiday', 'isFullMoon', 'weekend']
for column in columns_to_convert:
    crime[column] = crime[column].astype(bool)
columns_to_convert = ['Beat', 'Community.Area', 'windgust']
for column in columns_to_convert:
    crime[column] = crime[column].astype(float)



# %%
crime.dtypes

# %%
X_train, X_test, y_train, y_test = train_test_split(crime.drop('Crimes',axis=1), crime['Crimes'],random_state=101, train_size=.8)

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

# pipe = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', RandomForestRegressor(max_depth=150, random_state=101, n_jobs=-1))
# ])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# %%
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1)
])




# %%
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_squared_error'])


# %%
history = model.fit(X_train_preprocessed, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)


# %%
test_results = model.evaluate(X_test_preprocessed, y_test, verbose=1)
print(test_results)

###SHAP
# %%

#explainer = shap.DeepExplainer(model, X_train_preprocessed)

#shap_values = explainer.shap_values(X_test_preprocessed)
#shap_values_squeezed = np.array(shap_values[0]).squeeze(-1)
explainer = shap.Explainer(model, X_train_preprocessed)

# Calculate SHAP values for the test set
shap_values = explainer(X_test_preprocessed)

# Visualizing the first prediction
# SHAP values for the first instance are accessed with shap_values[0]
# Note: shap_values.values provides the raw SHAP values array
shap.force_plot(
    base_value=explainer.expected_value,  # or explainer.expected_value[0] if it's a list/array
    shap_values=shap_values.values[0],    # SHAP values for the first instance
    features=X_test_preprocessed[0]       # Actual feature values for the first instance
)

#%%
explainer = shap.Explainer(model, X_train_preprocessed)

# Calculate SHAP values for the test set
shap_values = explainer(X_test_preprocessed)

feature_names = preprocessor.get_feature_names_out()

# Then use these feature names with your SHAP force plot
shap.force_plot(
    shap_values[0],    # Accessing the first set of SHAP values for the first instance
    feature_names=feature_names,  # This now contains the correctly defined feature names
    matplotlib=True    # Optional: Use matplotlib=True to generate plots that are easier to display in some environments
)


# %%
