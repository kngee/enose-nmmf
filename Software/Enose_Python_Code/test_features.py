import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# File and label mapping
gas_files = {
    'Methanol': 'Methanol.csv',
    'Isopropyl': 'Isopropyl.csv',
    'Acetone': 'Acetone.csv',
    'Glycol': 'Glycol.csv',
}

# Load and label data
def load_and_label():
    dfs = []
    for label, file in gas_files.items():
        df = pd.read_csv(file)
        n_top = max(1, int(np.ceil(0.15 * len(df))))
        df = df.head(n_top)
        df['label'] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def plot_feature_distributions(features):
    plt.figure(figsize=(12, 4))
    for i, stat in enumerate(['mean', 'variance', 'skewness']):
        plt.subplot(1, 3, i+1)
        sns.boxplot(x='label', y=stat, data=features)
        plt.title(f'{stat.capitalize()} by Gas Type')
    plt.tight_layout()
    plt.show()

def plot_pairplot(features):
    sns.pairplot(features, hue='label', diag_kind='kde')
    plt.suptitle('Pairplot of Features', y=1.02)
    plt.show()

def plot_classifier_performance(results):
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    plt.figure(figsize=(8, 5))
    plt.bar(names, means, yerr=stds, color=sns.color_palette('viridis', len(names)))
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('Classifier Performance')
    plt.ylim(0, 1)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_sensor_ratios(features):
    ratio_cols = [col for col in features.columns if col.startswith('ratio_')]
    n = len(ratio_cols)
    n_per_fig = 4
    n_figs = (n + n_per_fig - 1) // n_per_fig
    for fig_idx in range(n_figs):
        start = fig_idx * n_per_fig
        end = min(start + n_per_fig, n)
        cols = ratio_cols[start:end]
        plt.figure(figsize=(5 * len(cols), 5))
        for i, col in enumerate(cols):
            plt.subplot(1, len(cols), i+1)
            sns.boxplot(x='label', y=col, data=features)
            plt.title(f'{col} by Gas Type')
            plt.ylabel('Sensor Ratio')
            plt.xlabel('Gas Type')
        plt.tight_layout()
        plt.show()

def plot_sensor_max_min(features, sensor_cols):
    # Plot max
    print('--- Sensor Maximum Value Comparison ---')
    n = len(sensor_cols)
    n_per_fig = 4
    n_figs = (n + n_per_fig - 1) // n_per_fig
    for fig_idx in range(n_figs):
        start = fig_idx * n_per_fig
        end = min(start + n_per_fig, n)
        cols = sensor_cols[start:end]
        plt.figure(figsize=(5 * len(cols), 5))
        for i, col in enumerate(cols):
            plt.subplot(1, len(cols), i+1)
            sns.boxplot(x='label', y=f'max_{col}', data=features)
            plt.title(f'Max {col} by Gas Type')
            plt.ylabel('Max Value')
            plt.xlabel('Gas Type')
        plt.tight_layout()
        plt.show()
    # Plot min
    print('--- Sensor Minimum Value Comparison ---')
    for fig_idx in range(n_figs):
        start = fig_idx * n_per_fig
        end = min(start + n_per_fig, n)
        cols = sensor_cols[start:end]
        plt.figure(figsize=(5 * len(cols), 5))
        for i, col in enumerate(cols):
            plt.subplot(1, len(cols), i+1)
            sns.boxplot(x='label', y=f'min_{col}', data=features)
            plt.title(f'Min {col} by Gas Type')
            plt.ylabel('Min Value')
            plt.xlabel('Gas Type')
        plt.tight_layout()
        plt.show()

def plot_sensor_gradients(features, sensor_cols):
    print('--- Sensor Gradient Comparison (Over Time) ---')
    grad_cols = [f'grad_{col}' for col in sensor_cols]
    n = len(grad_cols)
    n_per_fig = 4
    n_figs = (n + n_per_fig - 1) // n_per_fig
    for fig_idx in range(n_figs):
        start = fig_idx * n_per_fig
        end = min(start + n_per_fig, n)
        cols = grad_cols[start:end]
        plt.figure(figsize=(5 * len(cols), 5))
        for i, col in enumerate(cols):
            plt.subplot(1, len(cols), i+1)
            sns.boxplot(x='label', y=col, data=features)
            plt.title(f'Time Gradient {col[5:]} by Gas Type')
            plt.ylabel('Time Gradient')
            plt.xlabel('Gas Type')
        plt.tight_layout()
        plt.show()

def compute_features(df):
    feature_rows = []
    labels = []
    sensor_cols = [col for col in df.columns if col != 'label']
    for idx, row in df.iterrows():
        values = row[sensor_cols].values.astype(float)
        features = {
            'mean': np.mean(values),
            'variance': np.var(values),
            'skewness': skew(values),
            'label': row['label']
        }
        # Add max and min for each sensor (per sample, so just the value itself)
        for i, col in enumerate(sensor_cols):
            features[f'max_{col}'] = values[i]
            features[f'min_{col}'] = values[i]
        feature_rows.append(features)
        labels.append(row['label'])
    return pd.DataFrame(feature_rows), np.array(labels), sensor_cols

def main():
    df = load_and_label()
    # Compute global correlation matrix for reporting
    corr = df.drop('label', axis=1).corr()
    print('Global Feature Correlation Matrix:')
    print(corr)
    print('\n')
    # Compute per-sample features
    features, y, sensor_cols = compute_features(df)
    X = features.drop('label', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    classifiers = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    y_preds = {}
    print('--- Feature Distribution Comparison (Mean, Variance, Skewness) ---')
    plot_feature_distributions(features)
    plot_sensor_max_min(features, sensor_cols)
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
        results[name] = (scores.mean(), scores.std())
        y_preds[name] = y_pred
        print(f'{name} Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})')
    for name in classifiers:
        plot_confusion_matrix(y, y_preds[name], labels=list(gas_files.keys()), title=f'{name} Confusion Matrix')
    # Terminal summary
    best_model = max(results, key=lambda k: results[k][0])
    print('\nSummary:')
    print(f'The best classifier is {best_model} with an average accuracy of {results[best_model][0]:.3f}.')
    print('Boxplots show how well the features separate the classes.\n')
    print('Check the plots for visualizations and confusion matrices for detailed performance.')

if __name__ == '__main__':
    main()
