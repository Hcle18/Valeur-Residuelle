import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

class CategoricalEmbedding:
    """    Class to handle categorical embeddings and numerical features for a neural network model.
    
    This class prepares the data, creates the model architecture, and provides methods to visualize embeddings.
    It supports embedding for categorical features, one-hot encoding for other categorical features,
    and scaling for numerical features. The model architecture can be customized with hidden layers and dropout rates.
    Attributes:
        df (pd.DataFrame): Input DataFrame containing features and target.
        embed_features (list[str]): List of categorical features to be embedded.
        categorical_features (list[str]): List of other categorical features to be one-hot encoded.
        numerical_features (list[str]): List of numerical features to be scaled.
        embedding_dims (dict[str, int]): Optional dictionary specifying the embedding dimensions for each feature.
        target_column (str): Name of the target column in the DataFrame.
        hidden_layers (list[int]): List of integers specifying the number of units in each hidden layer.
        dropout_rates (list[float]): List of dropout rates for each hidden layer.    
    """
    def __init__(self, 
               df: pd.DataFrame,
               embed_features: list[str], # Cat Features to use embeddings
               categorical_features: list[str], # Other cat features
               numerical_features: list[str], # Num features
               embedding_dims: dict[str, int] = None, # Optional: embedding dimensions
               target_column: str = None, # Target column
               hidden_layers: list[int] = None, # Layer dimensions
               dropout_rates: list[float] = None # Dropout rates
               ):
        self.df = df
        self.embed_features = embed_features
        self.categorical_features = [f for f in categorical_features if f not in embed_features]
        self.numerical_features = numerical_features
        self.embedding_dims = embedding_dims or {f: 2 for f in embed_features}
        self.encoders = {}
        self.onehot_dim = {} # Store dimensions of one-hot encoded features
        self.target_column = target_column
        self.hidden_layers = hidden_layers or self._get_default_architecture()
        self.dropout_rates = dropout_rates or [0.3, 0.2, 0.1]
    
    def _get_default_architecture(self):
        total_dim = sum(self.embedding_dims.values()) + \
                    len(self.categorical_features) + \
                    len(self.numerical_features)
        return [
            max(64, total_dim),
            max(32, total_dim // 2),
            max(16, total_dim // 4)
        ]

    def prepare_data(self):
        """
        Prepare data with specified embeddings and other features
        """
        X = {}

        # process features that need embedding
        for feature in self.embed_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(self.df[feature]).reshape(-1, 1)
            self.encoders[feature] = le
            #print(self.encoders[feature].classes_)
        
        # process other cat features (one-hot encoding)
        for cat_feature in self.categorical_features:
            ohe = OneHotEncoder(sparse_output=False)
            encoded = ohe.fit_transform(self.df[[cat_feature]]) # binary arrays of shape (n_samples, n_categories)
            X[cat_feature] = encoded
            self.encoders[cat_feature] = ohe
            self.onehot_dim[cat_feature] = encoded.shape[1]

        # process numerical features
        if self.numerical_features:
            scaler = StandardScaler()
            numerical_data = self.df[self.numerical_features].copy()
            X['numerical_features'] = scaler.fit_transform(numerical_data)
            self.encoders['scaler'] = scaler

        # prepare target
        y = self.df[self.target_column].values

        return X, y
    
    def create_model(self):
        """
        Create model with specified embeddings and features
        """
        inputs = {}
        features = []

        # Create embedding layers
        for feature in self.embed_features:
            inp = keras.layers.Input(shape=(1,), name = feature)
            inputs[feature] = inp
            embedding = keras.layers.Embedding(
                input_dim = len(self.encoders[feature].classes_),
                output_dim = self.embedding_dims[feature],
                name = f"{feature}_embedding"
            )(inp)
            # Concatenate embedding layers
            features.append(keras.layers.Flatten()(embedding))

        # Create other categorical inputs (no embedding)
        for cat_feature in self.categorical_features:
            inp = keras.layers.Input(shape=(self.onehot_dim[cat_feature],), name=cat_feature)
            inputs[cat_feature] = inp
            features.append(inp)

        # Create numerical input
        num_inp = keras.layers.Input(shape=(len(self.numerical_features),),
                                     name = 'numerical_features')
        inputs['numerical_features'] = num_inp
        features.append(num_inp)

        # Combine all features
        combined = keras.layers.Concatenate()(features)

        # Dense layer: input layer, embedding layer and hidden layer(s)
        x = combined
        for units, dropout_rate in zip(self.hidden_layers, self.dropout_rates):
            x = keras.layers.Dense(units, activation = 'relu')(x)
            #x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(dropout_rate)(x)
        # Output layer
        output = keras.layers.Dense(1, activation = "linear")(x)

        print(keras.Model(inputs= inputs, outputs = output).summary())

        return keras.Model(inputs= inputs, outputs = output)

def plot_embeddings(model, feature_name, encoder, method='tsne'):
    # Get embedding weights
    embedding_layer = model.get_layer(f"{feature_name}_embedding")
    embeddings = embedding_layer.get_weights()[0]
    # # Create a DataFrame for embeddings
    # embeddings_df = pd.DataFrame(embeddings)
    # # Add labels
    # labels = encoder.classes_
    # # Set the index to the labels
    # embeddings_df.index = labels
    # # Add prefix to columns
    # embeddings_df = embeddings_df.add_prefix(feature_name + '_')
    # print(embeddings_df)

    # Reduce dimensionality to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    embeddings_2d = reducer.fit_transform(embeddings)

    # Create plot
    plt.figure(figsize=(12,8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    # Ajouter la droite 0, 0
    plt.axhline(0, color='white', lw=0.5, ls='--')
    plt.axvline(0, color='white', lw=0.5, ls='--')
    # Add labels
    for i, label in enumerate(encoder.classes_):
        plt.annotate(
            label,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext = (5,5),
            textcoords= 'offset points',
            fontsize = 8,
            alpha = 0.7
        )
    plt.title(f"{feature_name.capitalize()} Embeddings Visualization ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.show()

def visualize_all_embeddings(model_handler, model):
    """
    Visualize embeddings for all embedded features
    """
    for feature in model_handler.embed_features:
        print(f"\nVisualizing embeddings for {feature}:")

        # t-SNE visualization
        plot_embeddings(
            model, 
            feature, 
            model_handler.encoders[feature], 
            method='tsne'
        )
        # PCA visualization
        plot_embeddings(
            model, 
            feature, 
            model_handler.encoders[feature], 
            method='pca'
        )
        
# Get embedding weights
def get_embedding_weights(model, encoder, feature_name):
    """
    Get the embedding weights for a specific feature from the model.
    
    Args:
        model: The trained model.
        encoder: The Encoder used for the feature.
        feature_name: The name of the feature for which to get the embeddings.
    
    Returns:
        A DataFrame containing the embeddings for the specified feature.
    """
    embedding_layer = model.get_layer(f"{feature_name}_embedding")
    embeddings = embedding_layer.get_weights()[0]

    # Create a DataFrame for embeddings
    embeddings_df = pd.DataFrame(embeddings)
    # Add labels
    labels = encoder.classes_
    # Set the index to the labels
    embeddings_df.index = labels
    # Add prefix to columns
    embeddings_df = embeddings_df.add_prefix(feature_name + '_')
    #print(embeddings_df)

    return embeddings_df

# Regrouper en classes en se basant sur les embeddings (K-means)
def cluster_embeddings(df_embed, col_embed,  n_clusters, list_columns: list[str]):
    kmeans = KMeans(n_clusters = n_clusters, random_state=42)
    kmeans.fit(df_embed[list_columns])
    
    df_embed_cluster = df_embed.copy()
    df_embed_cluster['cluster_' + col_embed] = kmeans.labels_
    # Ajouter les coordonnées des centres des clusters
    centers = kmeans.cluster_centers_
    centers_df = pd.DataFrame(centers, columns=list_columns)
    # Renommer l'index pour ajouter les centres des clusters
    centers_df.index = ['center_' + str(i) for i in range(n_clusters)]
    centers_df['cluster_' + col_embed] = range(n_clusters)
    df_embed_cluster = pd.concat([df_embed_cluster, centers_df], ignore_index=False)

    # Ajouter un graphique pour visualiser les clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_embed_cluster, x=list_columns[0], y=list_columns[1], hue='cluster_' + col_embed, palette='viridis')
    # Ajouter les étiquettes d'Index
    for i in range(len(df_embed_cluster)):
        plt.text(df_embed_cluster.iloc[i][list_columns[0]], df_embed_cluster.iloc[i][list_columns[1]], 
                 df_embed_cluster.index[i], fontsize=9, alpha=0.7)
    plt.show()

    # Reset index, mettre l'index à la colonne col_embed
    df_embed_cluster.reset_index(inplace=True)
    df_embed_cluster.rename(columns={'index': col_embed}, inplace=True)
    return df_embed_cluster

if __name__ == "__main__":
    # Example
    # Obtenir le chemin du répertoire actuel et le chemin du répertoire racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Le chemin vers les données
    data_path = os.path.join(project_root, "data", "processed_data", "preprocessed_data.csv")
    df = pd.read_csv(data_path)
    print(df.head())

    embed_features = ['marque', 'modele']
    categorical_features = ['carburant', 'transmission', 'classe_vehicule', 'couleur']
    numerical_features = ['kilometrage', 'puissance', 'emission_CO2', 'age_months', 'prix_neuf']
    target_col = 'ratio_vr'

    # Optional: Custom embedding dimensions
    embedding_dims = {'marque': 2, 'modele': 2}

    # Create model instance
    model_handler = CategoricalEmbedding(
        df = df,
        embed_features=embed_features,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        embedding_dims=embedding_dims,
        target_column=target_col,
        hidden_layers = [128, 64, 32],
        dropout_rates= [0.2, 0.1, 0.1]
    )

    X, y = model_handler.prepare_data()

    # Split the data into training and validation sets
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42)
    print(len(train_idx))

    X_train_dict = {k: v[train_idx] for k, v in X.items()}
    X_val_dict = {k: v[test_idx] for k, v in X.items()}
    y_train = y[train_idx]
    y_val = y[test_idx]

    # Create and compile the model
    model = model_handler.create_model()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics =['mae'])

    # Train the model
    history = model.fit(X_train_dict, y_train, 
                        validation_data=(X_val_dict, y_val), 
                        epochs=10, batch_size=32)
    
    # Summary history for loss
    plt.plot(history.history['loss'], label ='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Summary history for MAE
    plt.plot(history.history['mae'], label ='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

    # Make predictions
    y_pred = model.predict(X_val_dict)
    y_pred = y_pred.flatten()
    ax = sns.scatterplot(x=y_val, y=y_pred)

    model_error = y_val - y_pred
    R2 = 1 - sum(model_error**2) / sum((y_val - np.mean(y_val))**2)
    RMSE = np.sqrt(np.mean(model_error**2))
    MAE = np.mean(np.abs(model_error))
    print(f"R2: {R2:.4f}, RMSE: {RMSE:.4f}, MAE: {MAE:.4f}")

    # Visualize embeddings
    visualize_all_embeddings(model_handler, model)
