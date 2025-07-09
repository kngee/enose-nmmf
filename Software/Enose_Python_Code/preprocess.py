import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_data(X, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, scaler, pca

