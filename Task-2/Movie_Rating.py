import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("Movie-Rating.csv", encoding='ISO-8859-1')

# Drop rows with missing target (Rating)
data = data.dropna(subset=['Rating'])

# Clean Data
data['Year'] = data['Year'].str.extract(r'(\d{4})')
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

data['Duration'] = data['Duration'].str.extract(r'(\d+)')
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')

data['Votes'] = data['Votes'].str.replace(',', '', regex=True)
data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')

# Drop columns that are not useful
data = data.drop(['Name'], axis=1)

# Select relevant columns
features = ['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1']
data = data[features + ['Rating']].dropna()

# Split input and output
X = data.drop(columns=['Rating'])
y = data['Rating']

# Categorical and numerical columns
categorical_cols = ['Genre', 'Director', 'Actor 1']
numerical_cols = ['Year', 'Duration', 'Votes']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep numerical features
)

# Full pipeline with Random Forest
model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_pipeline.fit(X_train, y_train)

# Predict
y_pred = model_pipeline.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # use square root manually
r2 = r2_score(y_test, y_pred)

# Output results
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
