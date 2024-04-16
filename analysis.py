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


#%%
from tensorflow.keras.callbacks import EarlyStopping

# Initialize the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='mean_absolute_error',   
    patience=15,           # Number of epochs with no improvement after which training will be stopped
    verbose=1,             # To log when training is stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored metric
)





# %%
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_squared_error'])


# %%
history = model.fit(
    X_train_preprocessed,
    y_train,
    validation_data=(X_test_preprocessed, y_test),
    epochs=100,
    callbacks=[early_stopping]
)


# %%
test_results = model.evaluate(X_test_preprocessed, y_test, verbose=1)
print(test_results)

###SHAP
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

##Psuedo R-Squared
#%%

# Generate predictions for the training and testing data
y_train_pred = model.predict(X_train_preprocessed).flatten()

# Calculate the residuals
residuals_train = y_train - y_train_pred

np.corrcoef(residuals_train,y_train)















# %%

plt.figure(figsize=(20, 10)) # Increase the figure size as needed

# Generate the SHAP force plot
shap.force_plot(
    shap_values[0].base_values, # Use the base value for the first instance
    shap_values[0].values,     # SHAP values for the first instance
    feature_names=feature_names,  # The correctly defined feature names
    matplotlib=True             # Generate the plot using matplotlib
)

# %%
shap_html = shap.force_plot(
    shap_values[0].base_values,  # Base value for the first instance
    shap_values[0].values,       # SHAP values for the first instance
    feature_names=feature_names,  # The correctly defined feature names
    show=False                    # Do not show the plot immediately since we want to save it
)

# Save the plot to an HTML file
shap.save_html('shap_force_plot.html', shap_html)

# Generate a textual summary
summary_text = "\n".join(f"{feature}: {value:.4f}" for feature, value in zip(feature_names, shap_values[0].values))

# Save the summary to a text file
with open('shap_summary.txt', 'w') as file:
    file.write(summary_text)

# Provide the paths to the saved HTML and summary text files
shap_plot_html_path = 'shap_force_plot.html'
shap_summary_text_path = 'shap_summary.txt'
# %%

# Create a horizontal bar plot with SHAP values
fig, ax = plt.subplots(figsize=(10, len(feature_names) / 2))  # Set an appropriate figure size

# Sort the features by their SHAP values
sorted_indices = np.argsort(shap_values[0].values)
sorted_feature_names = np.array(feature_names)[sorted_indices]
sorted_shap_values = shap_values[0].values[sorted_indices]

# Create the bar plot
ax.barh(range(len(sorted_feature_names)), sorted_shap_values, color='skyblue')

# Annotate the bars with the SHAP values
for index, value in enumerate(sorted_shap_values):
    ax.text(value, index, str(round(value, 4)), va='center', ha='right' if value < 0 else 'left')

# Set the y-ticks to feature names
ax.set_yticks(range(len(sorted_feature_names)))
ax.set_yticklabels(sorted_feature_names)

# Labeling
ax.set_xlabel('SHAP value')
ax.set_title('SHAP values per feature')

# Show grid
ax.grid(True)



# %%
filtered_indices = [i for i, feature in enumerate(feature_names) if not feature.endswith('^2')]
filtered_shap_values = shap_values[0].values[filtered_indices]
filtered_feature_names = np.array(feature_names)[filtered_indices]

# Now sort by absolute SHAP values and get the indices of the top 10
top_indices = np.argsort(-np.abs(filtered_shap_values))[:15]

# Get the SHAP values and names for the top 10 features
top_shap_values = filtered_shap_values[top_indices]
top_feature_names = filtered_feature_names[top_indices]

# Create a horizontal bar plot for the top 10 features
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
plt.barh(top_feature_names, top_shap_values, color='skyblue')

# Annotate the bars with SHAP values
for i, (value, feature) in enumerate(zip(top_shap_values, top_feature_names)):
    plt.text(value, i, f'{value:.4f}', va='center', ha='right' if value < 0 else 'left')

# Labeling the plot
plt.xlabel('SHAP value')
plt.title('Top 15 SHAP values per feature')
plt.grid(True)

# Show the plot
plt.show()
# %%
