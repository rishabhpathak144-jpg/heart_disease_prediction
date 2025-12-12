import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load dataset
df = pd.read_csv("heart.csv")

# 2. Features and target
X = df.drop("target", axis=1)
y = df["target"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Define and train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate model
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# 6. Save trained model to disk
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
