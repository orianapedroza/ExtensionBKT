import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union, Any, Optional
from models.re_bkt import REBKT
from utils.metrics import compute_metrics, plot_confusion_matrix

class ModelBase:
    """
    API para gestionar modelos RE-BKT particionados por clústeres.
    Permite entrenar, predecir y evaluar masivamente.
    """

    def __init__(self, seed: int = 42, cluster_col: str = 'cluster'):
        self.seed = seed
        self.cluster_col = cluster_col
        self.models = {}  # clave: (skill_name, cluster) -> REBKT

    def _prepare_sequences(self, skill_data: pd.DataFrame) -> List[List[Tuple[int, float]]]:
        """Convierte los datos de una habilidad (y opcionalmente un clúster) en una lista de secuencias
        de tuplas (correct, concentration)."""
        sequences = []
        for user_id, group in skill_data.groupby('user_id'):
            sequence = []
            for _, row in group.iterrows():
                correct = int(row['correct'])
                concentration = float(row['concentrating'])
                sequence.append((correct, concentration))
            sequences.append(sequence)
        return sequences

    def fit(self, data: pd.DataFrame):
        """
        Itera sobre cada habilidad y cada clúster en el dataset
        y entrena un modelo REBKT independiente para cada uno.
        """
        if self.cluster_col not in data.columns:
            raise ValueError(f"El DataFrame debe tener la columna '{self.cluster_col}'")

        unique_skills = data['skill'].unique()
        unique_clusters = data[self.cluster_col].unique()

        for skill_name in unique_skills:
            for cl in unique_clusters:
                # Filtrar datos de esta habilidad y este clúster
                mask = (data['skill'] == skill_name) & (data[self.cluster_col] == cl)
                skill_cluster_data = data[mask]

                if skill_cluster_data.empty: continue

                # Preparar secuencias por usuario
                observations_list = self._prepare_sequences(skill_cluster_data)

                # Solo entrenar si hay al menos una secuencia
                if len(observations_list) == 0:
                    continue

                #print(f"\n--- Entrenando para {skill_name} / clúster {cl} ---")
                rebkt = REBKT(skill_name=f"{skill_name}_cl{cl}", seed=self.seed)
                rebkt.fit(observations_list)

                self.models[(skill_name, cl)] = rebkt

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Asigna predicciones a un DataFrame de prueba basándose en la configuración 
        de pesos emocionales por clúster.

        """
        if self.cluster_col not in test_data.columns:
            raise ValueError(f"El DataFrame de prueba debe tener la columna '{self.cluster_col}'")

        #if cluster_config is None: cluster_config = {}

        df = test_data.copy()
        df['_original_index'] = range(len(df))
        # Ordenar para que cada usuario tenga sus interacciones consecutivas
        df_sorted = df.sort_values(by=['user_id', 'skill']).reset_index(drop=True)

        predictions_by_skill_cluster = {}  # clave (skill, cluster) -> lista de predicciones

        for (skill_name, cl), model in self.models.items():
            mask = (df_sorted['skill'] == skill_name) & (df_sorted[self.cluster_col] == cl)
            if not mask.any():
                continue

            #config = cluster_config.get(cl, {'emotion_weight': 1.0, 'neutral_point': 0.5})
            #ew = config.get('emotion_weight', 1.0)
            #np_val = config.get('neutral_point', 0.5)

            skill_cluster_data = df_sorted[mask]
            sequences = self._prepare_sequences(skill_cluster_data)

            if len(sequences) == 0:
                continue

            preds, actuals = model.predict(sequences)
            predictions_by_skill_cluster[(skill_name, cl)] = preds

        # Asignar predicciones al DataFrame ordenado
        df_sorted['prediction'] = np.nan

        for (skill_name, cl), preds in predictions_by_skill_cluster.items():
            mask = (df_sorted['skill'] == skill_name) & (df_sorted[self.cluster_col] == cl)
            idx = df_sorted[mask].index
            # El número de predicciones debe coincidir con el número de filas en esas posiciones
            if len(idx) != len(preds):
                pass
            df_sorted.loc[idx, 'prediction'] = preds

        df_sorted = df_sorted.sort_values(by='_original_index').reset_index(drop=True)
        return df_sorted.drop(columns=['_original_index'])
    
    def evaluate(self, data: pd.DataFrame, by_cluster: bool = False, verbose: bool = False) -> dict:
        
        if 'prediction' not in data.columns:
            raise ValueError("El DataFrame debe contener la columna 'prediction' generada por predict()")

        # Eliminar filas con predicción NaN
        original_len = len(data)
        data_clean = data.dropna(subset=['prediction']).copy()

        #print(f"DEBUG: Filas recibidas: {len(data)}")
        #print(f"DEBUG: Filas con predicción (sin NaN): {len(data_clean)}")
        #if thresholds is None: thresholds = {}

        if len(data_clean) < original_len:
            #print(f"Advertencia: Se eliminaron {original_len - len(data_clean)} filas con predicción NaN.")
            if len(data_clean) == 0:
                return {'global': {'Error': 'No hay predicciones válidas'}}

        if by_cluster:
            clusters = data_clean[self.cluster_col].unique()
            metrics_by_cluster = {}
            all_metrics = []

            for cl in clusters:
                cluster_data = data_clean[data_clean[self.cluster_col] == cl]
                y_true = cluster_data['correct'].tolist()
                y_pred = cluster_data['prediction'].tolist()
                model_sample = next((m for (s, c), m in self.models.items() if c == cl), None)
                current_threshold = model_sample.threshold if model_sample else 0.5
                #print(f"umbral: {current_threshold}", {cl})
                metrics = compute_metrics(y_true, y_pred, threshold=current_threshold)
                metrics_by_cluster[f"cluster_{cl}"] = metrics
                all_metrics.append(metrics)

            global_metrics = {}
            for key in metrics_by_cluster[list(metrics_by_cluster.keys())[0]]:
                global_metrics[key] = np.mean([m[key] for m in all_metrics])

            result = {'global': global_metrics, 'by_cluster': metrics_by_cluster}
            '''if verbose:
                print("\n--- Métricas globales ---")
                for k, v in global_metrics.items():
                    print(f"  {k}: {v:.4f}")
                for cl, met in metrics_by_cluster.items():
                    print(f"\n--- Métricas para {cl} ---")
                    for k, v in met.items():
                        print(f"  {k}: {v:.4f}")'''
            return result
        else:
            y_true = data_clean['correct'].tolist()
            y_pred = data_clean['prediction'].tolist()
            metrics = self._compute_metrics(y_true, y_pred, verbose=verbose)
            return {'global': metrics}

    def save_params(self, file_path: str):
        """
        Exporta los parámetros de todos los modelos entrenados (por habilidad y clúster)
        a un archivo CSV.
        """
        if not self.models:
            print("Advertencia: No hay modelos entrenados para guardar.")
            return

        rows = []
        for (skill_name, cluster_id), model in self.models.items():
            # Obtener parámetros del modelo REBKT
            params = model.get_params()
            if not params:
                continue

            # Añadir información de identificación
            row = {
                'skill': skill_name,
                'cluster': cluster_id
            }
            row.update(params)
            rows.append(row)

        df_params = pd.DataFrame(rows)
        df_params.to_csv(file_path, index=False)
        print(f"✓ Parámetros guardados exitosamente en: {file_path}")

    def load_params(self, params_source: Union[str, pd.DataFrame]):
        """
        Carga los parámetros desde un CSV y reconstruye los modelos REBKT
        en el diccionario self.models.
        """
        try:
            # 1. Determinar el origen de los datos
            if isinstance(params_source, str):
                df = pd.read_csv(params_source)
                source_name = params_source
            elif isinstance(params_source, pd.DataFrame):
                df = params_source.copy()
                source_name = "DataFrame proporcionado"
            else:
                raise TypeError("params_source debe ser una ruta (str) o un pd.DataFrame")
            
            expected_cols = ['skill', 'cluster', 'P_L0', 'P_T', 'P_IC0', 'P_CC0', 'P_IC1', 'P_CC1']

            if not all(col in df.columns for col in expected_cols):
                raise ValueError("El archivo CSV no tiene el formato correcto de parámetros RE-BKT.")

            self.models = {}
            for _, row in df.iterrows():
                skill_name = str(row['skill'])
                cluster_id = row['cluster']

                # Crear instancia de REBKT
                rebkt = REBKT(skill_name=f"{skill_name}_cl{cluster_id}", seed=self.seed)

                # Reconstruir el diccionario de parámetros
                rebkt.params = {
                    'P_L0': float(row['P_L0']),
                    'P_T':  float(row['P_T']),
                    'P_IC0': float(row['P_IC0']),
                    'P_CC0': float(row['P_CC0']),
                    'P_IC1': float(row['P_IC1']),
                    'P_CC1': float(row['P_CC1'])
                }
                rebkt.trained = True

                # Guardar en el diccionario de ModelBase
                self.models[(skill_name, cluster_id)] = rebkt

            print(f"✓ Se cargaron satisfactoriamente {len(self.models)} modelos desde {source_name}")

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en la ruta {params_source}")
        except Exception as e:
            print(f"Ocurrió un error al cargar los parámetros: {e}")

    def _assign_hyperparams_to_models(self, cluster: int, emotion_weight: float, neutral_point: float, threshold: float):
        """Asigna los hiperparámetros a todos los modelos de un cluster."""
        for (skill_name, cl), model in self.models.items():
            if cl == cluster:
                model.emotion_weight = emotion_weight
                model.neutral_point = neutral_point
                model.threshold = threshold        