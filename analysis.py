# %%
import pandas as pd

# %%
crime = pd.read_csv("crimes_cleaned.csv")
crime['Arrest'] = crime['Arrest'].astype(int)
crime['Domestic'] = crime['Domestic'].astype(int)
# %%
crime.head()