import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib

class EnoseClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.gas_names = None
        
    def load_data_from_files(self, file_paths_dict, has_header=False):
        """
        Load sensor data from multiple CSV files (one per gas)
        
        Parameters:
        file_paths_dict: dictionary mapping gas names to file paths
                        e.g., {'Methanol': 'methanol_data.csv', 'Acetone': 'acetone_data.csv'}
        has_header: whether the CSV files have header rows
        """
        X_list = []
        y_list = []
        
        for gas_name, file_path in file_paths_dict.items():
            print(f"Loading {gas_name} data from {file_path}...")
            
            try:
                if has_header:
                    df = pd.read_csv(file_path)
                    # Use only numeric columns
                    X_gas = df.select_dtypes(include=[np.number]).values
                else:
                    df = pd.read_csv(file_path, header=None)
                    # Try to convert all columns to numeric, non-convertible values become NaN
                    df_numeric = df.apply(pd.to_numeric, errors='coerce')
                    # Keep columns where at least 90% of values are numeric
                    col_mask = (df_numeric.notnull().mean() > 0.9)
                    df_numeric = df_numeric.loc[:, col_mask]
                    # Drop all rows with any NaN in the selected columns
                    df_numeric = df_numeric.dropna(axis=0, how='any')
                    X_gas = df_numeric.values
                
                # Add this gas's data
                X_list.append(X_gas)
                y_list.extend([gas_name] * len(X_gas))
                
                print(f"  Loaded {len(X_gas)} samples for {gas_name}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        # Combine all data
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        # Set feature names
        self.feature_names = [f'Sensor_{i+1}' for i in range(X.shape[1])]
        
        print(f"\nTotal dataset: {len(X)} samples, {X.shape[1]} sensors")
        print(f"Gas distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for gas, count in zip(unique, counts):
            print(f"  {gas}: {count} samples")
        
        # Encode gas labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.gas_names = self.label_encoder.classes_
        
        return X, y_encoded
    
    def load_data(self, data_path=None, X=None, y=None, has_header=False):
        """
        Load sensor data either from file or directly from arrays
        
        Parameters:
        data_path: path to CSV file with sensor data
        X: numpy array of sensor readings (n_samples, n_sensors)
        y: numpy array of gas labels
        has_header: whether the CSV file has a header row
        """
        if data_path:
            # Load from CSV file
            if has_header:
                df = pd.read_csv(data_path)
                # Assuming last column is the gas label
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                self.feature_names = df.columns[:-1].tolist()
            else:
                # No header, just sensor data
                df = pd.read_csv(data_path, header=None)
                X = df.values
                self.feature_names = [f'Sensor_{i+1}' for i in range(X.shape[1])]
                y = None
        elif X is not None:
            # Use provided arrays
            X = np.array(X)
            if y is not None:
                y = np.array(y)
            self.feature_names = [f'Sensor_{i+1}' for i in range(X.shape[1])]
        else:
            raise ValueError("Either provide data_path or X array")
        
        # If no labels provided, return just X
        if y is None:
            print("No gas labels provided. You'll need to add labels for training.")
            return X, None
        
        # Encode gas labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.gas_names = self.label_encoder.classes_
        
        return X, y_encoded
    
    def add_labels_to_data(self, X, labels_per_segment):
        """
        Add gas labels to unlabeled sensor data
        
        Parameters:
        X: sensor data array
        labels_per_segment: list of tuples [(gas_name, n_samples), ...]
        
        Example:
        labels_per_segment = [('Methane', 50), ('CO', 50), ('Hydrogen', 50), ('Ammonia', 50)]
        """
        y = []
        current_idx = 0
        
        for gas_name, n_samples in labels_per_segment:
            if current_idx + n_samples > len(X):
                raise ValueError(f"Not enough samples for {gas_name}. Need {n_samples}, but only {len(X) - current_idx} remaining.")
            
            y.extend([gas_name] * n_samples)
            current_idx += n_samples
        
        if current_idx < len(X):
            print(f"Warning: {len(X) - current_idx} samples left unlabeled")
            
        y = np.array(y)
        y_encoded = self.label_encoder.fit_transform(y)
        self.gas_names = self.label_encoder.classes_
        
        return X[:current_idx], y_encoded
    
    def preprocess_data(self, X_train, X_test=None):
        """
        Standardize the sensor data
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train SVM classifier with hyperparameter tuning
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Preprocess data
        X_train_scaled, X_test_scaled = self.preprocess_data(X_train, X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly'],
            'probability': [True]  # Ensure predict_proba is available
        }
        
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            SVC(), 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest accuracy: {accuracy:.4f}")
        
        # Store test results for visualization
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.y_pred = y_pred
        
        return self.model
    
    def evaluate_model(self):
        """
        Print detailed evaluation metrics
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Ensure gas_names is a list of strings
        gas_names_list = [str(g) for g in self.gas_names] if self.gas_names is not None else []
        print("\nClassification Report:")
        print(classification_report(
            self.y_test, 
            self.y_pred, 
            target_names=gas_names_list
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=gas_names_list, 
                   yticklabels=gas_names_list)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def visualize_data(self, X, y):
        """
        Visualize sensor data using PCA
        """
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'orange']
        gas_names_list = [str(g) for g in self.gas_names] if self.gas_names is not None else []
        for i, gas in enumerate(gas_names_list):
            mask = y == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colors[i % len(colors)], label=gas, alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('E-nose Data Visualization (PCA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def predict_new_sample(self, sensor_readings):
        """
        Predict gas type for new sensor readings
        
        Parameters:
        sensor_readings: array-like, sensor values for a single sample
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        if self.gas_names is None:
            raise ValueError("Gas names not set!")
        # Reshape if single sample
        sensor_readings = np.array(sensor_readings)
        if len(sensor_readings.shape) == 1:
            sensor_readings = sensor_readings.reshape(1, -1)
        # Check feature count
        if sensor_readings.shape[1] != self.scaler.n_features_in_:
            raise ValueError(f"Input has {sensor_readings.shape[1]} features, but model expects {self.scaler.n_features_in_}.")
        # Scale the input
        sensor_readings_scaled = self.scaler.transform(sensor_readings)
        # Predict
        prediction = self.model.predict(sensor_readings_scaled)
        if prediction is None or not hasattr(prediction, '__getitem__') or len(prediction) == 0:
            raise ValueError("Prediction failed: model did not return a result.")
        probabilities = self.model.predict_proba(sensor_readings_scaled)
        gas_name = self.label_encoder.inverse_transform([prediction[0]])[0]
        print(f"Predicted gas: {gas_name}")
        print("Confidence scores:")
        gas_names_list = [str(g) for g in self.gas_names]
        for i, gas in enumerate(gas_names_list):
            print(f"  {gas}: {probabilities[0][i]:.4f}")
        return gas_name, probabilities[0]
    
    def save_model(self, filename):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'gas_names': self.gas_names
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load a trained model"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.gas_names = model_data['gas_names']
        print(f"Model loaded from {filename}")

def filter_top_percent_per_gas(X, y, gas_names, percent=0.2):
    """
    Filter to the top X% of data samples for each gas class, based on the sum of sensor values for each sample.
    Args:
        X: Feature matrix
        y: Encoded labels
        gas_names: List of gas names (as in classifier.gas_names)
        percent: Fraction of samples to keep for each gas (e.g., 0.2 for top 20%)
    Returns:
        X_out, y_out: Filtered feature matrix and labels
    """
    X_filtered = []
    y_filtered = []
    for gas_idx, gas in enumerate(gas_names):
        mask = y == gas_idx
        X_gas = X[mask]
        n_top = max(1, int(np.ceil(len(X_gas) * percent)))
        # Use sum of sensor values as the ranking metric
        scores = X_gas.sum(axis=1)
        # Select the top n_top samples (highest values)
        top_indices = np.argsort(scores)[-n_top:]
        X_filtered.append(X_gas[top_indices])
        y_filtered.append(np.full(n_top, gas_idx))
    X_out = np.vstack(X_filtered)
    y_out = np.concatenate(y_filtered)
    return X_out, y_out

def kfold_evaluate(classifier, X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    all_accuracies = []
    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"\n=== Fold {fold} ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Preprocess
        X_train_scaled, X_test_scaled = classifier.preprocess_data(X_train, X_test)
        # Train
        model = SVC(C=1, gamma=1, kernel='rbf', probability=True)  # Use best params or tune here
        model.fit(X_train_scaled, y_train)
        # Predict
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=[str(g) for g in classifier.gas_names]))
        all_accuracies.append(acc)
        fold += 1
    print(f"\nMean accuracy over {k} folds: {np.mean(all_accuracies):.4f}")

# Example usage with your separate CSV files
if __name__ == "__main__":
    print("="*70)
    print("E-NOSE CLASSIFIER FOR METHANOL, GLYCOL, ACETONE, ISOPROPYL")
    print("="*70)
    
    # Create classifier instance
    classifier = EnoseClassifier()
    
    print("\n" + "="*60)
    print("LOADING DATA FROM SEPARATE CSV FILES")
    print("="*60)
    
    # Update file paths to match your actual CSVs
    file_paths = {
        'Methanol': 'Methanol.csv',
        'Glycol': 'Glycol.csv', 
        'Acetone': 'Acetone.csv',
        'Isopropyl': 'Isopropyl.csv'
    }
    
    # Load data from files (no headers in your CSVs)
    X, y = classifier.load_data_from_files(file_paths, has_header=False)
    # Filter to top 20% of data samples for each gas
    X, y = filter_top_percent_per_gas(X, y, classifier.gas_names, percent=0.2)
    
    print(f"Dataset loaded successfully! (Top 20% of samples for each gas by sum of sensor values)")
    print(f"Total samples: {len(X)}")
    print(f"Sensors: {len(X[0])}")
    print(f"Gas types: {classifier.gas_names}")
    
    print("\n" + "="*60)
    print("K-FOLD CROSS-VALIDATION")
    print("="*60)
    kfold_evaluate(classifier, X, y, k=5)
    
    # Visualize the data distribution
    print("\n" + "="*60)
    print("DATA VISUALIZATION")
    print("="*60)
    classifier.visualize_data(X, y)
    
    # Train the model
    print("\n" + "="*60)
    print("TRAINING SVM CLASSIFIER")
    print("="*60)
    model = classifier.train_model(X, y)
    
    # Evaluate model performance
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    classifier.evaluate_model()

    # Save the model
    classifier.save_model('enose_SVM_classifier.pkl')