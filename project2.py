# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ------------------------------
# Step 1: Load & Preprocess Data
# ------------------------------
# Load dataset (update path if needed)
file_path = "../data/test_small.csv"  # Using sampled dataset
df = pd.read_csv(file_path)

# Drop unnecessary columns (modify as per your dataset)
df.drop(columns=["Item_Identifier", "Outlet_Identifier"], inplace=True, errors="ignore")

# Handle missing values
df.fillna(df.median(), inplace=True)

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# -----------------------------------------
# Step 2: Autoencoder + K-Means
# -----------------------------------------
# Build Autoencoder
input_dim = df_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
latent = Dense(2, activation='relu')(encoded)  # 2D latent space
decoded = Dense(32, activation='relu')(latent)
decoded = Dense(64, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')
autoencoder.fit(df_scaled, df_scaled, epochs=100, batch_size=32, verbose=0)

# Extract latent features
encoder = Model(input_layer, latent)
latent_features = encoder.predict(df_scaled)

# Apply K-Means on latent features
kmeans = KMeans(n_clusters=4, random_state=42)
labels_ae = kmeans.fit_predict(latent_features)

# Evaluate
sil_ae = silhouette_score(latent_features, labels_ae)
db_ae = davies_bouldin_score(latent_features, labels_ae)
print(f"\nAutoencoder + K-Means: Silhouette = {sil_ae:.2f}, Davies-Bouldin = {db_ae:.2f}")

# -----------------------------------------
# Step 3: Deep Embedded Clustering (DEC)
# -----------------------------------------
class DEC:
    def __init__(self, n_clusters, input_dim):
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.autoencoder = autoencoder  # Reuse pretrained autoencoder
        self.model = self.build_model()
        self.model.compile(optimizer=Adam(0.001), loss='kld')

    def build_model(self):
        # Get latent layer from autoencoder
        latent_output = self.autoencoder.layers[-4].output  
        # Add clustering layer
        clustering_layer = Dense(self.n_clusters, activation='softmax')(latent_output)
        return Model(inputs=self.autoencoder.input, outputs=clustering_layer)

    def predict_clusters(self, X):
        q = self.model.predict(X, verbose=0)
        return q.argmax(1)

# Initialize and train DEC
dec = DEC(n_clusters=4, input_dim=input_dim)
# Use cluster predictions as training targets
dec.model.fit(df_scaled, dec.model.predict(df_scaled), epochs=150, verbose=0)
labels_dec = dec.predict_clusters(df_scaled)

# Evaluate
sil_dec = silhouette_score(df_scaled, labels_dec)
db_dec = davies_bouldin_score(df_scaled, labels_dec)
print(f"DEC: Silhouette = {sil_dec:.2f}, Davies-Bouldin = {db_dec:.2f}")

# -----------------------------------------
# Step 4: Visualization (t-SNE)
# -----------------------------------------
# Reduce dimensions for plotting
tsne = TSNE(n_components=2, random_state=42)
df_tsne = tsne.fit_transform(df_scaled)

# Plot clusters
plt.figure(figsize=(15, 5))

# Autoencoder + K-Means
plt.subplot(1, 2, 1)
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=labels_ae, cmap='viridis', s=10)
plt.title("Autoencoder + K-Means Clusters")

# DEC
plt.subplot(1, 2, 2)
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=labels_dec, cmap='viridis', s=10)
plt.title("DEC Clusters")

plt.tight_layout()
plt.show()