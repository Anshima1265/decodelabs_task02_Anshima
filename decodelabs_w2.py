from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("=" * 50)
print("DATA CLASSIFICATION USING AI")
print("=" * 50)

print(f"\nTotal Samples: {len(X)}")
print(f"Total Features: {len(feature_names)}")

print("\nFlower Classes:")
for flower in target_names:
    print("-", flower)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Successfully")

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("Model Trained Successfully")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL RESULTS")
print("=" * 50)

print(f"\nAccuracy Score: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

importance = model.feature_importances_

plt.figure(figsize=(8, 5))

plt.bar(feature_names, importance)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.xticks(rotation=20)

plt.tight_layout()
plt.show()

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

plt.figure(figsize=(10, 5))

plt.plot(
    comparison.index,
    comparison["Actual"],
    marker='o',
    label="Actual"
)

plt.plot(
    comparison.index,
    comparison["Predicted"],
    marker='x',
    label="Predicted"
)

plt.title("Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Class")
plt.legend()

plt.grid(True)

plt.show()

print("\n" + "=" * 50)
print("PREDICT YOUR OWN FLOWER")
print("=" * 50)

try:

    sepal_length = float(input("Enter Sepal Length (cm): "))
    sepal_width = float(input("Enter Sepal Width (cm): "))
    petal_length = float(input("Enter Petal Length (cm): "))
    petal_width = float(input("Enter Petal Width (cm): "))

    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(sample)

    flower_name = target_names[prediction[0]]

    print("\nPrediction Result:")
    print("Flower Type =", flower_name.upper())

except ValueError:
    print("Please enter valid numbers.")

print("\nProject Completed Successfully!")
