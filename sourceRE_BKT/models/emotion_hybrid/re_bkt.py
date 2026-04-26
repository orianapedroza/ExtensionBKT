#EMOCIÓN INTERNA Y EXTERNA

import numpy as np
from typing import List, Tuple, Dict
from .baum_welch import BaumWelchREBKT


class REBKT:
    """
    Modelo RE-BKT para una habilidad específica.
    Utiliza una matriz de observación B de dimensión 2×4:
        0: (Incorrecto, Concentrado)    → IC
        1: (Incorrecto, No Concentrado) → INC
        2: (Correcto,   Concentrado)    → CC
        3: (Correcto,   No Concentrado) → CNC

    La predicción de P(correcto) incorpora la concentración continua
    mediante una transformación logística.
    """

    def __init__(self, skill_name: str, seed: int = 42):
        self.skill_name = skill_name
        self.seed = seed
        self.trained = False
        self.params = None
        self.model_base = None
        self.emotion_weight = 1.0
        self.neutral_point = 0.5
        self.threshold = 0.5

    def fit(self, observations_list: List[List[Tuple[int, float]]], num_restarts: int = 1):
        """
        Entrena el modelo usando Baum-Welch con múltiples reinicios
        para evitar mínimos locales.
        """
        best_ll = -np.inf
        best_model = None

        for restart in range(num_restarts):
            tmp_bw = BaumWelchREBKT()
            tmp_bw.fit(observations_list, self.threshold ) # Llama al fit matemático

            if len(tmp_bw.history) > 0:
                final_ll = tmp_bw.history[-1]
                if final_ll > best_ll:
                    best_ll = final_ll
                    best_model = tmp_bw

        if best_model is None:
            best_model = tmp_bw

        self.baum_welch = best_model
        self.params = self.baum_welch.get_params()
        self.trained = True
        #print(f"\n✓ RE-BKT entrenado. Mejor Log-Likelihood: {best_ll:.4f}")


    def predict(self, test_sequences: List[List[Tuple[int, float]]], emotion_weight=None, neutral_point=None) -> Tuple[List[float], List[int]]:
        """
        Realiza predicciones sobre secuencias de estudiantes.
        Implementa la lógica de actualización Bayesiana + Regresión Logística.
        """
        ew = emotion_weight if emotion_weight is not None else self.emotion_weight
        np_val = neutral_point if neutral_point is not None else self.neutral_point
        threshold = getattr(self, 'threshold', 0.5)

        if not self.trained:
            raise ValueError(f"El modelo para {self.skill_name} no ha sido entrenado.")

        P_L0 = self.params['P_L0']
        P_T  = self.params['P_T']

        # Mapa de emision
        b0 = {
            0: self.params['P_IC0'],   # IC
            1: self.params['P_INC0'],  # INC
            2: self.params['P_CC0'],   # CC
            3: self.params['P_CNC0'],  # CNC
        }
        b1 = {
            0: self.params['P_IC1'],
            1: self.params['P_INC1'],
            2: self.params['P_CC1'],
            3: self.params['P_CNC1'],
        }

        # Probabilidad base de correcto = suma de obs correctas (CC + CNC)
        P_correct_NDom = b0[2] + b0[3]  # P_CC0 + P_CNC0
        P_correct_Dom  = b1[2] + b1[3]  # P_CC1 + P_CNC1

        all_predictions, all_actuals = [], []

        for d, sequence in enumerate(test_sequences):
            P_L = P_L0

            for t, (obs_y, c_t) in enumerate(sequence):
                # Probabilidad base BKT
                P_correct_base = P_L * P_correct_Dom + (1 - P_L) * P_correct_NDom

                # Regresión logística con parámetros optimizados
                p_safe = np.clip(P_correct_base, 0.01, 0.99)
                logit_base = np.log(p_safe / (1 - p_safe))
                logit_final = logit_base + (ew * (c_t - np_val))
                P_correct = 1 / (1 + np.exp(-logit_final))

                all_predictions.append(P_correct)
                all_actuals.append(obs_y)

                # Discretizo con el umbral optimizado la emocion
                is_concentrated = (c_t >= threshold)

                if obs_y == 1:
                    obs_idx = 2 if is_concentrated else 3 # PCC (2) o PCNC (3)
                else:
                    obs_idx = 0 if is_concentrated else 1 # IC (0) o INC (1)

                # Actualización y Transición
                numerator = P_L * b1[obs_idx]
                denominator = P_L * b1[obs_idx] + (1 - P_L) * b0[obs_idx]
                P_L_after_obs = numerator / denominator if denominator > 0 else P_L
                P_L = P_L_after_obs + (1 - P_L_after_obs) * P_T

        return all_predictions, all_actuals

    def get_params(self) -> dict:
        return self.params if self.params else {}