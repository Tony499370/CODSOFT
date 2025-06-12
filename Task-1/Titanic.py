import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('Titanic-Dataset.csv')

# Drop columns with string data that are not useful for prediction
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Encode categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Handle missing values (fill with median for simplicity)
data = data.fillna(data.median(numeric_only=True))

x = data.drop(['Survived'], axis=1)
y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
best_n_estimators = 10  

best_score = 0
scores_list = []    
for n_estimators in range(10, 201, 10):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=42))
    ])
    scores = cross_val_score(pipeline, x_train, y_train, cv=5)
    scores_list.append(scores.mean())
    
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_n_estimators = n_estimators

print(f"Best n_estimators: {best_n_estimators} with cross-validation accuracy: {best_score:.2f}")

# Train final model using best n_estimators
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=best_n_estimators, random_state=42))
])

final_pipeline.fit(x_train, y_train)

# Predict and evaluate
y_pred = final_pipeline.predict(x_test)

print("\nFinal Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot accuracy vs n_estimators
plt.figure(figsize=(8, 5))
plt.plot(range(10, 201, 10), scores_list, marker='o')
plt.title("Cross-Validated Accuracy vs Number of Trees (n_estimators)")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Cross-Validated Accuracy")
plt.grid(True)
plt.savefig('accuracy_vs_n_estimators.png')