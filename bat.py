import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')
print(df.describe())
X = df.drop('target_column', axis=1) # Replace 'target_column' with your RUL column
y = df['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Model selection
model = RandomForestRegressor()

model.fit(X_train, y_train)



