import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ======================
# 1. Load Features
# ======================
X = np.load(f"{ROOT}/outputs/X_train.npy")
y = np.load(f"{ROOT}/outputs/y_train.npy")
groups = np.load(f"{ROOT}/outputs/groups_train.npy")

# ======================
# 2. Cross-Validation & Model Selection
# ======================
gkf = GroupKFold(n_splits=5)
models = {
    "logistic": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight='balanced'))]),
    "naive_bayes": GaussianNB(),
    "decision_tree": DecisionTreeClassifier(max_depth=12),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=gkf.split(X, y, groups))
    print(f"{name} CV Accuracy: {scores.mean():.4f}")
    cv_results[name] = scores.mean()
    model.fit(X, y)
    joblib.dump(model, f"{ROOT}/models/{name}_model.pkl")

# ======================
# 3. Train Committee Models
# ======================
best_model_name = max(cv_results, key=cv_results.get)
print("Best model:", best_model_name)

for i in range(5):
    students_unique = np.unique(groups)
    sample_students = np.random.choice(students_unique, size=int(0.8*len(students_unique)), replace=True)
    idx = [j for j, g in enumerate(groups) if g in sample_students]

    if best_model_name == "logistic":
        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    elif best_model_name == "naive_bayes":
        model = GaussianNB()
    elif best_model_name == "decision_tree":
        model = DecisionTreeClassifier(max_depth=12)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)

    model.fit(X[idx], y[idx])
    joblib.dump(model, f"{ROOT}/models/committee_model_{i}.pkl")

# ======================
# 4. Learning Curve
# ======================
train_sizes = np.linspace(0.1, 1.0, 10)
accuracies = []
unique_students = np.unique(groups)

for frac in train_sizes:
    n_students = int(frac * len(unique_students))
    selected_students = np.random.choice(unique_students, n_students, replace=False)
    mask = np.isin(groups, selected_students)
    X_subset = X[mask]
    y_subset = y[mask]
    groups_subset = groups[mask]

    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    scores = cross_val_score(model, X_subset, y_subset, cv=GroupKFold(5).split(X_subset, y_subset, groups_subset))
    accuracies.append(scores.mean())

plt.figure()
plt.plot(train_sizes, accuracies, marker='o')
plt.xlabel("Training Size (%)")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.grid()
plt.savefig(f"{ROOT}/outputs/figures/learning_curve.png")
plt.show()

# ======================
# 5. Model Comparison Table
# ======================
cv_table = pd.DataFrame({"Model": list(cv_results.keys()), "Accuracy": list(cv_results.values())})
print(cv_table)