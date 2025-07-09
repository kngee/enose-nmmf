import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from preprocess import preprocess_data

# Keep in mind that we need to concatenate the data. 
df = pd.read_csv("data/mq_sensor_voc_data.csv") 
X = df.drop("Gas_Type", axis=1)
y = df["Gas_Type"]

# Preprocess
X_pca, scaler, pca = preprocess_data(X, n_components=3)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train RF model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save models
joblib.dump(clf, "models/rf_model.pkl")
joblib.dump(pca, "models/pca_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Training complete. Models saved.")
