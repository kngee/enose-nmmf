import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

class ENoseKNNClassifier:
    def __init__(self, n_neighbors=5, n_folds=5):
        """
        Initialize the E-nose KNN classifier
        
        Parameters:
        n_neighbors: Number of neighbors for KNN (default: 5)
        n_folds: Number of folds for cross-validation (default: 5)
        """
        self.n_neighbors = n_neighbors
        self.n_folds = n_folds
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def prepare_data(self, X, y):
        """
        Prepare the data by scaling features and encoding labels
        
        Parameters:
        X: Feature matrix (sensor readings)
        y: Target labels (gas types)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded
    
    def optimize_k(self, X, y, k_range=range(1, 21)):
        """
        Find optimal k value using cross-validation
        
        Parameters:
        X: Feature matrix
        y: Target labels
        k_range: Range of k values to test
        """
        X_scaled, y_encoded = self.prepare_data(X, y)
        
        k_scores = []
        for k in k_range:
            knn_temp = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn_temp, X_scaled, y_encoded, 
                                   cv=self.n_folds, scoring='accuracy')
            k_scores.append(scores.mean())
        
        # Find optimal k
        optimal_k = k_range[np.argmax(k_scores)]
        
        # Plot k optimization
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, k_scores, 'bo-')
        plt.axvline(x=optimal_k, color='r', linestyle='--', 
                   label=f'Optimal k = {optimal_k}')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('KNN k-Value Optimization')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"Optimal k value: {optimal_k}")
        print(f"Best cross-validation accuracy: {max(k_scores):.4f}")
        
        # Update the classifier with optimal k
        self.n_neighbors = optimal_k
        self.knn = KNeighborsClassifier(n_neighbors=optimal_k)
        
        return optimal_k, k_scores
    
    def perform_cross_validation(self, X, y):
        """
        Perform k-fold cross-validation
        
        Parameters:
        X: Feature matrix
        y: Target labels
        """
        X_scaled, y_encoded = self.prepare_data(X, y)
        
        # Perform cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.knn, X_scaled, y_encoded, 
                                   cv=kf, scoring='accuracy')
        
        print(f"\n{self.n_folds}-Fold Cross-Validation Results:")
        print(f"Accuracy scores: {cv_scores}")
        print(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Standard deviation: {cv_scores.std():.4f}")
        
        return cv_scores
    
    def train_and_evaluate(self, X, y, test_size=0.2):
        """
        Train the model and evaluate on test set
        
        Parameters:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of data to use for testing
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Prepare training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Train the model
        self.knn.fit(X_train_scaled, y_train_encoded)
        self.is_fitted = True
        
        # Prepare test data
        X_test_scaled = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Make predictions
        y_pred = self.knn.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # Generate classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y_test_encoded, y_pred, 
                                     target_names=target_names)
        
        print(f"\nTest Set Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return accuracy, report, cm
    
    def predict_new_sample(self, X_new):
        """
        Predict gas type for new sensor readings
        
        Parameters:
        X_new: New sensor readings (can be single sample or multiple samples)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first!")
        
        # Ensure X_new is 2D
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        
        # Scale the new data
        X_new_scaled = self.scaler.transform(X_new)
        
        # Make prediction
        y_pred_encoded = self.knn.predict(X_new_scaled)
        
        # Get probabilities
        probabilities = self.knn.predict_proba(X_new_scaled)
        
        # Decode labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred, probabilities

def load_csv_data(csv_files, gas_names=None):
    """
    Load e-nose data from CSV files, using only the top 15% of each file
    
    Parameters:
    csv_files: List of CSV file paths (one per gas type)
    gas_names: List of gas names (if None, will use filenames)
    
    Returns:
    X: Combined feature matrix
    y: Combined labels array
    """
    X_data = []
    y_data = []
    
    if gas_names is None:
        gas_names = [f"Gas_{i+1}" for i in range(len(csv_files))]
    
    print("Loading CSV files (top 15% rows only)...")
    
    for i, (csv_file, gas_name) in enumerate(zip(csv_files, gas_names)):
        try:
            # Load CSV file
            df = pd.read_csv(csv_file)
            # Remove any non-numeric columns (like timestamps, labels, etc.)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            data = df[numeric_columns].to_numpy()
            # Only keep the top 15% of the rows
            n_rows = data.shape[0]
            n_top = max(1, int(np.ceil(n_rows * 0.15)))
            data_top = data[:n_top]
            print(f"Loaded {gas_name}: {data_top.shape[0]} samples (top 15%), {data_top.shape[1]} features")
            # Add to dataset
            X_data.append(data_top)
            y_data.extend([gas_name] * data_top.shape[0])
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    # Combine all data
    X = np.vstack(X_data)
    y = np.array(y_data)
    print(f"\nCombined dataset:")
    print(f"Total samples: {X.shape[0]}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Gas types: {np.unique(y)}")
    return X, y

def run_enose_classification_from_csvs(csv_files, gas_names=None):
    """
    Complete e-nose classification pipeline using CSV files
    
    Parameters:
    csv_files: List of CSV file paths
    gas_names: List of gas names (optional)
    """
    print("E-nose KNN Classification from CSV Files")
    print("=" * 50)
    
    # Load data from CSV files (top 20% only)
    X, y = load_csv_data(csv_files, gas_names)
    
    if X.size == 0:
        print("No data loaded. Please check your CSV files.")
        return
    
    # PCA plot (2D)
    print("\nPCA visualization of the dataset (top 20% rows per file)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    for gas in np.unique(y):
        idx = y == gas
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=gas, alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of E-nose Sensor Data (Top 20% per file)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Initialize classifier
    classifier = ENoseKNNClassifier(n_neighbors=5, n_folds=5)
    
    # Optimize k value
    print("\n1. Optimizing k value...")
    optimal_k, k_scores = classifier.optimize_k(X, y)
    
    # Perform cross-validation
    print("\n2. Performing cross-validation...")
    cv_scores = classifier.perform_cross_validation(X, y)
    
    # Train and evaluate
    print("\n3. Training and evaluating model...")
    accuracy, report, cm = classifier.train_and_evaluate(X, y)
    
    return classifier, X, y

# Example usage with CSV files
def demo_csv_classification():
    """
    Example of how to use the classifier with your CSV files
    """
    # Use the actual CSV file paths present in the workspace
    csv_files = [
        'Methanol.csv',
        'Acetone.csv',
        'Isopropyl.csv',
        'Glycol.csv'
    ]
    
    # Specify gas names corresponding to the files
    gas_names = ['Methanol', 'Acetone', 'Isopropyl', 'Glycol']
    
    # Run classification
    classifier, X, y = run_enose_classification_from_csvs(csv_files, gas_names)
    
    # Example: Predict on a new sample
    if classifier is not None:
        print("\n4. Example prediction...")
        # Use the first sample from the dataset as an example
        sample_idx = 0
        test_sample = X[sample_idx]
        actual_label = y[sample_idx]
        
        prediction, probabilities = classifier.predict_new_sample(test_sample)
        
        print(f"Test sample (actual: {actual_label}):")
        print(f"Predicted: {prediction[0]}")
        print("Probabilities:")
        for i, gas in enumerate(np.unique(y)):
            prob_idx = list(classifier.label_encoder.classes_).index(gas)
            print(f"  {gas}: {probabilities[0][prob_idx]:.4f}")

if __name__ == "__main__":
    # Run the CSV-based classification
    demo_csv_classification()