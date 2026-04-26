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
        np.random.seed(42)
        self.filtered_raw_df = None

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
            skill_user_counts = df.groupby(['user_id', 'skill']).size()
            valid_combos = skill_user_counts[skill_user_counts >= 6].index
            if len(valid_combos) == 0:
                filtered_df = pd.DataFrame(columns=df.columns)
            else:
                valid_df = pd.DataFrame(valid_combos.tolist(), columns=['user_id', 'skill'])
                filtered_df = df.merge(valid_df, on=['user_id', 'skill'])
                # Mostrar información del filtrado
            print(f"Global: se conservaron {filtered_df['user_id'].nunique()} usuarios, {filtered_df['skill'].nunique()} habilidades.")
       
        elif strategy == 'sampled':
            # Caso 2: 25 estudiantes por habilidad, solo sus primeras 6 interacciones
            def sample_logic(skill_group):
                user_counts = skill_group.groupby('user_id').size()
                valid_users = user_counts[user_counts == 6].index
                n_valid = len(valid_users)
                
                if n_valid < 25:
                    return pd.DataFrame(columns=skill_group.columns)  # vacío
                else:
                    # Seleccionar aleatoriamente 25 usuarios
                    selected = np.random.choice(valid_users, size=25, replace=False)
                    return skill_group[skill_group['user_id'].isin(selected)]
            
            # Aplicar a cada skill y concatenar resultados no vacíos
            filtered_dfs = []
            for skill, group in df.groupby('skill'):
                skill_subset = sample_logic(group)
                if not skill_subset.empty:
                    filtered_dfs.append(skill_subset)
            
            if not filtered_dfs:
                filtered_df = pd.DataFrame(columns=df.columns)
            else:
                filtered_df = pd.concat(filtered_dfs, ignore_index=True)
                
            # Verificar que efectivamente cada skill tiene 25 usuarios y cada usuario 6 filas
            if not filtered_df.empty:
                print("Muestreo completado. Skills incluidos:")
                for skill, group in filtered_df.groupby('skill'):
                    n_users = group['user_id'].nunique()
                    # Verificar que cada usuario tiene 6 filas
                    rows_per_user = group.groupby('user_id').size()
                    assert all(rows_per_user == 6), f"Error: Usuario en skill {skill} no tiene exactamente 6 filas."
                    print(f"  Skill '{skill}': {n_users} usuarios (cada uno con 6 interacciones)")

        else:
            raise ValueError("Estrategia no reconocida. Use 'global' o 'sampled'")

        self.filtered_raw_df = filtered_df.copy() if not filtered_df.empty else None

        if filtered_df.empty:
            print("Advertencia: No hay datos después del filtrado.")
            return pd.DataFrame()         

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