# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
crime = pd.read_csv("crimes_cleaned.csv")
crime['Arrest'] = crime['Arrest'].astype(int)
crime['Domestic'] = crime['Domestic'].astype(int)
# %%
crime.head()

# %%
X_train, X_test, y_train, y_test = train_test_split(crime.drop('Primary Type',axis=1), crime['Primary Type'],random_state=101, train_size=.8)

# %%
X_train