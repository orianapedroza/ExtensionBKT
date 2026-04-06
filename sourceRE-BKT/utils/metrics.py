import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, roc_auc_score, accuracy_score,
                             f1_score, precision_score, recall_score, confusion_matrix)
from typing import List, Dict, Optional

def compute_metrics(y_true: List[int], 
                           y_pred: List[float], 
                           threshold: float = 0.5) -> Dict[str, float]:
    """
    Calcula un set completo de métricas para evaluación de BKT.
    
    Args:
        y_true: Etiquetas reales (0 o 1).
        y_pred: Probabilidades predichas por el modelo.
        threshold: Umbral para binarizar la predicción.
        
    Returns:
        Diccionario con Precision, Sensitivity, Specificity, Accuracy, RMSE, F1 y AUC.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    # Métricas basadas en probabilidad
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    try:
        auc = roc_auc_score(y_true_arr, y_pred_arr)
    except ValueError:
        auc = np.nan

    # Métricas basadas en clasificación (binarias)
    y_pred_bin = (y_pred_arr >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true_arr, y_pred_bin)
    precision = precision_score(y_true_arr, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_arr, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_bin, zero_division=0)

    # Especificidad (TN / (TN + FP))
    cm = confusion_matrix(y_true_arr, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'Precision': precision, 'Sensitivity': recall, 'Specificity': specificity,
        'Accuracy': accuracy, 'RMSE': rmse, 'F1_Score': f1, 'AUC': auc
    }

def plot_confusion_matrix(y_true: List[int], 
                          y_pred: List[float], 
                          threshold: float = 0.5, 
                          title: str = "Matriz de Confusión",
                          filename: Optional[str] = None):
    """
    Genera y guarda un mapa de calor de la matriz de confusión.
    """
    y_pred_bin = (np.array(y_pred) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Incorrecto', 'Correcto'], 
                yticklabels=['Incorrecto', 'Correcto'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(title)
    
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()