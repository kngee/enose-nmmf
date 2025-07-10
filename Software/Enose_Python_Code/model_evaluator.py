import os
import glob
import pandas as pd
from   sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load and label data from Training folder ---
data_dir = "Training"
all_files = glob.glob(os.path.join(data_dir, "*.csv"))

dfs = []
for file in all_files:
    # Extract gas name from filename (e.g., "acetone-2023.csv" -> "acetone")
    filename = os.path.basename(file)
    gas_name = filename.split("-")[0]
    
    # Load and label
    df = pd.read_csv(file)
    df["Gas_Type"] = gas_name
    dfs.append(df)

print(dfs)

# Concatenate all gas-labeled data
df = pd.concat(dfs, ignore_index=True)

# ---The rest of your pipeline continues here---
X = df.drop("Gas_Type", axis=1)
y = df["Gas_Type"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Train/test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# (Rest of your model_configs and evaluate_model remains unchanged)

# Model configurations
model_configs = {
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5]
        }
    },
    "SVM": {
        "model": SVC(),
        "params": {
            "C": [0.1, 1],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    }
}

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# --- 📊 PCA scatter plot ---
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10', s=50)
plt.title("2D PCA Projection of VOC Sensor Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Gas Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Train/test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)

def evaluate_model(name, model, param_grid, X_train, X_test, label):
    print(f"\n🔍 {name} ({label})")
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Results
    print(f"Best Params: {grid.best_params_}")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=y.unique(), yticklabels=y.unique()) # type: ignore
    plt.title(f"{name} - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# Run both PCA and non-PCA versions
for name, cfg in model_configs.items():
    evaluate_model(name, cfg["model"], cfg["params"], X_train_raw, X_test_raw, label="Raw Features")
    evaluate_model(name, cfg["model"], cfg["params"], X_train_pca, X_test_pca, label="PCA Features")
