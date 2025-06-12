import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


# Load data
data = pd.read_csv('iris.csv')
X = data.drop('species', axis=1)
y = data['species']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Find best K using cross-validation
best_k = 1
best_score = 0
scores_list = []

for k in range(1, 21):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    scores_list.append(scores.mean())
    
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_k = k

print(f"Best k: {best_k} with cross-validation accuracy: {best_score:.2f}")

# Train final model using best k
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])
final_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = final_pipeline.predict(X_test)

print("\nFinal Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot accuracy vs k
plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), scores_list, marker='o')
plt.title("Cross-Validated Accuracy vs K")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Cross-Validated Accuracy")
plt.grid(True)

plt.savefig("accuracy_vs_k.png")
