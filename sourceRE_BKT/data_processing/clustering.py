from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.stats import pointbiserialr
import pandas as pd
import numpy as np
import warnings

class StudentClustered:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.emotions = ['frustrated', 'confused', 'concentrating', 'bored']

    def extract_features(self, df):
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        features = []
        for user, data in df.groupby('user_id'):
            if len(data) < 5: continue
            
            row = {'user_id': user, 'mean_correct': data['correct'].mean()}
            for emo in self.emotions:
                row[f'mean_{emo}'] = data[emo].mean()
                row[f'std_{emo}'] = data[emo].std()
                try:
                    corr, _ = pointbiserialr(data['correct'], data[emo])
                    row[f'corr_{emo}'] = corr
                except:
                    row[f'corr_{emo}'] = np.nan
            features.append(row)
        return pd.DataFrame(features)

    def run_clustering(self, feature_df):
        # Lógica de Imputer -> Scaler -> KMeans que ya tienes
        cols = [c for c in feature_df.columns if c != 'user_id']
        X = SimpleImputer(strategy='mean').fit_transform(feature_df[cols])
        X_scaled = StandardScaler().fit_transform(X)
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        feature_df['cluster'] = kmeans.fit_predict(X_scaled)
        return feature_df, X_scaled
    
    def assign_and_split(self, df_original, feature_df, test_size=0.2):
        # 1. Mapeo de clústeres
        user_cluster = dict(zip(feature_df['user_id'], feature_df['cluster']))
        df_original['cluster'] = df_original['user_id'].map(user_cluster)
        
        # 2. Limpieza de nulos (estudiantes con < 5 interacciones)
        df_clean = df_original.dropna(subset=['cluster']).copy()
        df_clean['cluster'] = df_clean['cluster'].astype(int)
        
        # 3. División por usuarios (Train/Test Split)
        from sklearn.model_selection import train_test_split
        users = df_clean['user_id'].unique()
        train_u, test_u = train_test_split(users, test_size=test_size, random_state=42)
        
        train_df = df_clean[df_clean['user_id'].isin(train_u)]
        test_df = df_clean[df_clean['user_id'].isin(test_u)]
        
        return train_df, test_df