import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Step 1: Create Sample Dataset (if you don't have CSV)
# -----------------------------
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 5, 6, 7, 8, 1, 4],
    'attendance': [50, 60, 65, 70, 75, 80, 85, 90, 55, 60, 78, 82, 88, 92, 48, 72],
    'previous_score': [40, 45, 50, 55, 60, 65, 70, 75, 42, 48, 62, 68, 72, 78, 38, 58],
    'pass': [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

# -----------------------------
# Step 2: Features & Target
# -----------------------------
X = df[['study_hours', 'attendance', 'previous_score']]
y = df['pass']

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 6: Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Step 7: User Input Prediction
# -----------------------------
print("\n--- Student Performance Prediction ---")

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance (%): "))
previous_score = float(input("Enter previous score: "))

new_data = np.array([[study_hours, attendance, previous_score]])
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Prediction: PASS ✅")
else:
    print("Prediction: FAIL ❌")