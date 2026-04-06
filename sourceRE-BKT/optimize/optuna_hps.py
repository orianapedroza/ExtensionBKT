import optuna
import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix

# Imports internos de tu biblioteca
from fit.model_base import ModelBase
from utils.metrics import compute_metrics

def optimize_hyperparameters(model_base: ModelBase, 
                             test_data: pd.DataFrame, 
                             n_trials: int = 100, 
                             metric_to_optimize: str = 'AUC') -> None:
    """
    Optimiza emotion_weight, neutral_point y threshold para cada cluster presente en el conjunto de datos del test.
    Los valores óptimos se asignan directamente a los modelos de cada cluster en model_base.

    Parámetros:
    ----------
    model_base : ModelBase
        Modelo base ya entrenado (con modelos REBKT para cada skill y cluster).
    test_data : pd.DataFrame
        DataFrame de validación con columnas: skill, user_id, correct, concentrating, cluster.
    n_trials : int
        Número de intentos de Optuna por cluster.
    metric_to_optimize : str
        Métrica a maximizar: 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'geometric_mean'.
    """
    
    # 1. Validación de columnas necesarias
    required_cols = ['skill', 'user_id', 'correct', 'concentrating', model_base.cluster_col]
    for col in required_cols:
        if col not in test_data.columns:
            raise ValueError(f"test_data debe tener la columna '{col}'")

    clusters = test_data[model_base.cluster_col].unique()
    print(f"\n[Optuna] Iniciando optimización para clusters: {clusters}")

    for cl in clusters:
        print(f"\n>>> Optimizando Cluster: {cl}")
        
        # Filtrar datos y modelos específicos de este cluster
        cluster_val = test_data[test_data[model_base.cluster_col] == cl].copy()
        skills_in_cluster = cluster_val['skill'].unique()
        
        # Recuperamos solo los modelos que pertenecen a este cluster
        cluster_models = {}
        for skill in skills_in_cluster:
            key = (skill, cl)
            if key in model_base.models:
                cluster_models[skill] = model_base.models[key]
            else:
                print(f"Advertencia: No hay modelo para skill '{skill}' en cluster {cl}. Se omitirá en la optimización.")
                
        if not cluster_models:
            print(f"No hay modelos entrenados para el cluster {cl}. Saltando...")
            continue

        # Función objetivo para Optuna
        def objective(trial):
            # Sugerir hiperparámetros para este intento (trial)
            ew = trial.suggest_float('emotion_weight', 0.5, 3.0)
            np_val = trial.suggest_float('neutral_point', 0.0, 1.0)
            thr = trial.suggest_float('threshold', 0.0, 1.0)

            all_y_true = []
            all_y_pred = []

            # Evaluar todos los modelos de este cluster con los parámetros sugeridos
            for skill, model in cluster_models.items():
                skill_data = cluster_val[cluster_val['skill'] == skill]
                if skill_data.empty: continue

                # Preparar secuencias usando el método interno de ModelBase
                sequences = model_base._prepare_sequences(skill_data)
                if len(sequences) == 0: continue

                # Predecir con los hiperparámetros temporalmente
                preds, actuals = model.predict(sequences, emotion_weight=ew, neutral_point=np_val)
                all_y_pred.extend(preds)
                all_y_true.extend(actuals)

            if len(all_y_true) == 0:
                return 0.0  # Sin datos, métrica nula

            # Cálculo de métricas para Optuna
            y_true_arr = np.array(all_y_true)
            y_pred_arr = np.array(all_y_pred)
            
            try:
                auc = roc_auc_score(y_true_arr, y_pred_arr) if len(np.unique(y_true_arr)) > 1 else 0.5
            except:
                auc = 0.5

            # Métricas binarias basadas en el threshold sugerido
            y_pred_bin = (y_pred_arr >= thr).astype(int)
            sens = recall_score(y_true_arr, y_pred_bin, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_bin, labels=[0,1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = precision_score(y_true_arr, y_pred_bin, zero_division=0)

            # Selección de métrica objetivo
            if metric_to_optimize == 'AUC': return auc
            if metric_to_optimize == 'Sensitivity': return sens
            if metric_to_optimize == 'Specificity': return spec
            if metric_to_optimize == 'Precision': return prec
            if metric_to_optimize == 'geometric_mean':
                return (auc * sens * spec) ** (1/3) # Media geométrica de AUC, sensibilidad y especificidad
            
            return auc # Default

        # Ejecutar el estudio de Optuna para este cluster
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Asignar los mejores valores encontrados al ModelBase
        best_trial = study.best_trial
        best_ew = best_trial.params['emotion_weight']
        best_np = best_trial.params['neutral_point']
        best_thr = best_trial.params['threshold']
        best_score = best_trial.value
        
        print(f"\nMejores hiperparámetros para cluster {cl}:")
        print(f"  emotion_weight = {best_ew:.4f}")
        print(f"  neutral_point  = {best_np:.4f}")
        print(f"  threshold      = {best_thr:.4f}")
        print(f"  Mejor {metric_to_optimize} = {best_score:.4f}")
        print(f"Cluster {cl} optimizado: {metric_to_optimize}={study.best_value:.4f}")

        # Asignar los hiperparámetros a todos los modelos de este cluster
        model_base._assign_hyperparams_to_models(cl, best_ew, best_np, best_thr)
    print("\n[Optuna] Optimización finalizada exitosamente.")