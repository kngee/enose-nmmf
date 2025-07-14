import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for percentile calculation

# --- Load and label data from Training folder ---
data_dir = "Training"
all_files = glob.glob(os.path.join(data_dir, "*.csv"))

filtered_dfs = []
for file in all_files:
    # Extract gas name from filename (e.g., "acetone-2023.csv" -> "acetone")
    filename = os.path.basename(file)
    gas_name = filename.split("-")[0]
    
    # Load data for the current gas
    df_gas = pd.read_csv(file)
    df_gas["Gas_Type"] = gas_name
    
    # Step 1: Characterize each row by a maximum sensor reading
    # We assume all columns except 'Gas_Type' are sensor readings
    sensor_cols = [col for col in df_gas.columns if col != 'Gas_Type']
    df_gas['row_max_sensor_reading'] = df_gas[sensor_cols].max(axis=1)
    
    # Step 2: Determine the threshold for the top 20% maximum values for this gas type
    # Calculate the 80th percentile (to get the cutoff for the top 20%)
    if not df_gas.empty:
        threshold = np.percentile(df_gas['row_max_sensor_reading'], 80)
        
        # Filter the DataFrame to keep only rows where the row_max_sensor_reading
        # is greater than or equal to the calculated threshold.
        filtered_df_gas = df_gas[df_gas['row_max_sensor_reading'] >= threshold].copy()
        
        # Drop the temporary 'row_max_sensor_reading' column before adding to the list
        filtered_df_gas = filtered_df_gas.drop(columns=['row_max_sensor_reading'])
        
        filtered_dfs.append(filtered_df_gas)
        print(f"Gas: {gas_name}, Original rows: {len(df_gas)}, Filtered rows (top 20% max): {len(filtered_df_gas)}")
    else:
        print(f"Warning: {gas_name} dataframe is empty, skipping.")


# Concatenate all filtered gas-labeled data
if filtered_dfs:
    df = pd.concat(filtered_dfs, ignore_index=True)
else:
    print("No data left after filtering. Please check your input CSVs and filtering logic.")
    exit() # Exit if no data to process

print(f"\nTotal number of samples after filtering for top 20% max sensor reading per gas type: {len(df)}")
print(f"Distribution of Gas Types after filtering:\n{df['Gas_Type'].value_counts()}")

# --- The rest of your pipeline continues here ---
X = df.drop("Gas_Type", axis=1)
y = df["Gas_Type"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=3) # Still use 3 for overall variance, but plot 2D
X_pca = pca.fit_transform(X_scaled)

# Train/test split
# This split now applies to the data that has already been filtered to top 20% max values.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)


# Model configurations
model_configs = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42), # Added random_state for reproducibility
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5]
        }
    },
    "SVM": {
        "model": SVC(random_state=42), # Added random_state for reproducibility
        "params": {
            "C": [0.1, 1],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    },
    "KNN": { # KNN does not have random_state
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    }
}

# --- 📊 PCA scatter plot ---
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10', s=50)
plt.title("2D PCA Projection of VOC Sensor Data (Top 20% Max Values)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Gas Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


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
                xticklabels=y.unique(), yticklabels=y.unique())
    plt.title(f"{name} - {label} (Confusion Matrix)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# Run both PCA and non-PCA versions
for name, cfg in model_configs.items():
    evaluate_model(name, cfg["model"], cfg["params"], X_train_raw, X_test_raw, label="Raw Features")
    evaluate_model(name, cfg["model"], cfg["params"], X_train_pca, X_test_pca, label="PCA Features")