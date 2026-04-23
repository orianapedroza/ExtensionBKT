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

    def extract_features(self, df, strategy='global'):
        """
        Extrae características según dos estrategias:
        - 'global': Filtra estudiantes con al menos 6 interacciones totales.
        - 'sampled': Filtra las primeras 6 interacciones de solo 25 estudiantes por habilidad.
        """
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # Filtrado según estrategia
        if strategy == 'global':
            # Caso 1: Estudiantes con al menos 6 interacciones
            counts = df.groupby('user_id').size()
            valid_users = counts[counts >= 6].index
            filtered_df = df[df['user_id'].isin(valid_users)]
        elif strategy == 'sampled':
            # Caso 2: 25 estudiantes por habilidad, solo sus primeras 6 interacciones
            def sample_logic(skill_group):
                # Obtener 25 usuarios únicos de esta habilidad
                users = skill_group['user_id'].unique()
                selected_users = users[:25] # O usar np.random.choice para aleatoriedad
                
                # De esos usuarios, tomar solo sus primeras 6 interacciones
                subset = skill_group[skill_group['user_id'].isin(selected_users)]
                return subset.groupby('user_id').head(6)
                
            filtered_df = df.groupby('skill', group_keys=False).apply(sample_logic)

        # Calculo de características
        features = []
        for user, data in filtered_df.groupby('user_id'):
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
        
        # 2. Limpieza de nulos 
        df_clean = df_original.dropna(subset=['cluster']).copy()
        df_clean['cluster'] = df_clean['cluster'].astype(int)
        
        # 3. División por usuarios (Train/Test Split)
        from sklearn.model_selection import train_test_split
        users = df_clean['user_id'].unique()
        train_u, test_u = train_test_split(users, test_size=test_size, random_state=42)
        
        train_df = df_clean[df_clean['user_id'].isin(train_u)]
        test_df = df_clean[df_clean['user_id'].isin(test_u)]
        
        return train_df, test_df