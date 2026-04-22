# RE-BKT: EXTENSIÓN DEL “BAYESIAN KNOWLEDGE TRACING” PARA LA INCORPORACIÓN DE EMOCIONES 

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-success.svg)

## Descripción del Proyecto

Este repositorio contiene el código fuente y los notebooks del proyecto de grado titulado *"Extensión del Bayesian Knowledge Tracing (BKT) para la incorporación de emociones"*. El objetivo principal es desarrollar y validar el modelo **RE-BKT (Respuesta Emocional - BKT)**, una extensión del modelo clásico BKT que integra el estado emocional del estudiante como variable informativa en la inferencia del dominio de habilidades.

### Características principales

- **Segmentación de perfiles emocionales:** Mediante clustering no supervisado (K-Means), se identifican dos perfiles estudiantiles basados en emociones auto-reportadas: concentración, frustración, aburrimiento y confusión.

- **Incorporación de la concentración:** La variable de concentración se integra en la función de probabilidad de respuesta correcta mediante una transformación logística, mejorando la capacidad predictiva.

- **Personalización por clúster:** Los hiperparámetros de modulación emocional (peso emocional, punto neutro, umbral de decisión) se optimizan independientemente para cada perfil mediante Optuna.

- **Validación en datasets públicos:** Los experimentos se realizan sobre los conjuntos de datos **ASSISTments 2012-2013** y **ASSISTments Challenge 2017**, ampliamente utilizados en minería de datos educativos.

>Se usa como modelo base la biblioteca [pyBKT](https://github.com/CAHLR/pyBKT).

Puedes revisar también la [Guia_REBKT](https://colab.research.google.com/drive/1V93SsmfToIp_438C270sSf1eQGkOMhH-?usp=sharing)

---

## Requisitos del Sistema

Para reproducir los experimentos, se recomienda el uso de **Google Colab** o un entorno local con al menos 12 GB de RAM, debido al volumen de datos de ASSISTments.

Las dependencias principales incluyen:

* Python >= 3.8
* `pandas`, `numpy` (Manipulación de datos)
* `scikit-learn`, `scipy` (Clustering y pruebas estadísticas)
* `pyBKT` (Modelo BKT base de comparación)
* `optuna` (Optimización de hiperparámetros)
* `matplotlib`, `seaborn` (Visualización)

---
## Arquitectura del Repositorio

El proyecto está dividido en el código fuente (herramientas) y los cuadernos de experimentación (análisis):

```text
ExtensionBKT/
├── notebooks/                                  # Cuadernos de experimentación y manuales
│   ├── ASSISTments_Data_Preparation.ipynb      # Carga, limpieza y clustering de usuarios
│   └── Evaluacion_Base_PyBKT.ipynb             # Entrenamiento y evaluación (PyBKT vs RE-BKT)
├── sourceRE_BKT/                               # Código fuente de la biblioteca RE-BKT
│   ├── data_processing/                        # EDA
│   │   ├── __init__.py
│   │   ├── clustering.py                       # Identificación de perfiles con K-Means
│   │   └── data_loader.py                      # Limpieza para ASSISTments 2012/2017
│   ├── fit/                                    # Controladores de Entrenamiento
│   │   ├── __init__.py
│   │   └── model_base.py                       # Clase integradora
│   ├── models/                                 # Implementación matemática 
│   │   ├── __init__.py
│   │   ├── baum_welch.py                       # Ajuste de Expectativa-Maximización (EM)
│   │   └── re_bkt.py                           # Definición del modelo extendido con emociones
│   ├── optimize/                               # Tuning de hiperparámetros
│   │   ├── __init__.py
│   │   └── optuna_hps.py                       # Optimización de umbrales y métricas vía Optuna
│   ├── test/                                   
│   └── utils/                                  # Herramientas auxiliares y estadísticas
│       ├── __init__.py
│       └── metrics.py                          # Precisión, Sensibilidad, Especificidad, Exactitud, RMSE, F1-Score, AUC
├── README.md                                   

```
---
## Instalación y Configuración

```
!pip install optuna seaborn scikit-learn gdown
```

**Configurar el entorno en Colab/Jupiter:**

```
import os
import sys

USER = "orianapedroza"
REPO = "ExtensionBKT"
BRANCH = "master"

if not os.path.exists(REPO):
    !git clone -b {BRANCH} https://github.com/{USER}/{REPO}.git
else:
    print(f"El repositorio {REPO} ya existe. Actualizando...")
    !cd {REPO} && git pull origin {BRANCH}

ruta_biblioteca = f'/content/{REPO}/sourceRE_BKT'

if ruta_biblioteca not in sys.path:
    sys.path.append(ruta_biblioteca)

print("Entorno configurado: Biblioteca RE-BKT lista para usar.")

```
---
## Datasets Utilizados

Este proyecto procesa datos de interacción de las siguientes competiciones públicas:

* **ASSISTments 2009-2010 / 2012-2013:** https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010/data

* **ASSISTments 2017:** Datos anonimizados de la competencia oficial. https://sites.google.com/view/assistmentsdatamining/dataset

*(Nota: Los archivos CSV originales no están incluidos en este repositorio por su gran tamaño. Deben descargarse y colocarse en el entorno de ejecución antes de iniciar).*

---
## Preparación de Datos y Ejecución del Modelo

El siguiente es un guía sobre cómo empezar con RE-BKT. Los formatos de entrada aceptados son archivos de datos de ASSISTments (2012 o 2017).

### 1. Procesamiento de los datos
RE-BKT incluye un módulo robusto para homogenizar columnas, manejar valores nulos y filtrar inconsistencias temporales automáticamente.

```
from data_processing import DataLoader, StudentClustered

# Normalizar y Filtrar datos (ASSISTments 2017)
loader = DataLoader()
df_clean = loader.clean_2017('ruta/a/ASSISTments_2017.csv')

```
### 2. Identificación de Perfiles Emocionales (Clustering)

A diferencia del BKT tradicional, RE-BKT agrupa a los estudiantes basándose en sus respuestas emocionales utilizando K-Means y validación por correlación Punto-Biserial.

```
clusterer = StudentClustered(n_clusters=2)

# Caracterizar estudiantes y aplicar K-Means
feature_df = clusterer.extract_features(df_clean)
feature_df, X_scaled = clusterer.run_clustering(feature_df)

# Dividir el dataset por usuario (Train 80% / Test 20%)
train_df, test_df = clusterer.assign_and_split(df_clean, feature_df, test_size=0.2)
```

---
### 3. Entrenamiento (Algoritmo Baum Welch)

El núcleo de la biblioteca permite entrenar los modelos separando automáticamente las lógicas por clúster e integrando la optimización de hiperparámetros.

```
from fit import ModelBase

# Inicializar y ajustar el modelo usando el set de entrenamiento
modelo = ModelBase(seed=42, cluster_col='cluster')
modelo.fit(train_df)

```

---
### 4. Predicción

*Podemos asignar los valores a los hiperparámetros si ya los conocemos*
```
from optimize import optimize_hyperparameters

modelo._assign_hyperparams_to_models(cluster=0, emotion_weight=0.9071, neutral_point=0.7970, threshold=0.5282)
modelo._assign_hyperparams_to_models(cluster=1, emotion_weight=2.3075, neutral_point=0.3342, threshold=0.8582)
```

```
# Predecir probabilidades de conocimiento
predicciones = modelo.predict(test_df)
```

---
### 5. Evaluación y Análisis Estadístico (Utils)

El módulo de utilidades incluye herramientas diseñadas para la evaluación rigurosa del rendimiento predictivo (Precisión, Sensibilidad, Especificidad, Exactitud, RMSE, F1-Score, AUC) frente a BKT tradicional.

```
from utils import plot_confusion_matrix

metricas = modelo.evaluate(predicciones, by_cluster=True)

df_eval = predicciones.dropna(subset=['prediction'])

clusters_unicos = df_eval['cluster'].unique()

for cl in clusters_unicos:
    datos_cluster = df_eval[df_eval['cluster'] == cl]
    ejemplo_skill = test_df[test_df['cluster'] == cl]['skill'].iloc[0]
    threshold_guardado = modelo.models[(ejemplo_skill, cl)].threshold

    print(f"Generando matriz para Clúster {cl}...")
    plot_confusion_matrix(
        y_true=datos_cluster['correct'],
        y_pred=datos_cluster['prediction'],
        threshold=threshold_guardado,
        title=f"Matriz RE-BKT - Clúster {cl}",
        filename=f"matriz_cluster_{cl}.png"
    )
```
---
## Autoría

* **T.S.U Oriana Pedroza Palomar** - pedrozapalomar.ov07@gmail.com
* **Tutor: Dr. Jesús Pérez** 
* **Cotutor: Dr. José Aguilar** 

PROYECTO DE GRADO
Presentado ante la ilustre Universidad de Los Andes como requisito para obtener el título de Ingeniero de Sistemas

---

## Referencias y Citación

El desarrollo de esta arquitectura se basa en investigaciones previas de BKT, específicamente:

* M. Baktashmotlagh, Y. Wang, and K. R. Koedinger, "pyBKT: An accessible Python library for Bayesian Knowledge Tracing," in Proceedings of the 14th International Conference on Educational Data Mining (EDM 2021), I. Roll, D. McNamara, S. Sosnovsky, and R. Luckin, Eds., International Educational Data Mining Society, 2021, pp. 518–524.

* O. Pedroza, "Extensión del 'Bayesian Knowledge Tracing' para la incorporación de emociones," Trabajo especial de grado, Universidad de Los Andes, Mérida, Venezuela, mayo 2026.

*Si utilizas esta biblioteca para investigación académica, por favor cita los repositorios y autores correspondientes.*

---