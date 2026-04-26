import numpy as np
from typing import List, Tuple, Dict
from .baum_welch import BaumWelchREBKT

class REBKT:
    """
    Representa el modelo BKT Extendido para una habilidad específica.
    Maneja el entrenamiento (fit) y la inferencia (predict).
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
            tmp_bw.fit(observations_list) # Llama al fit matemático
            
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

        #print(f"ew: {ew}")
        #print(f"np = {np_val}")

        if not self.trained:
            raise ValueError(f"El modelo para {self.skill_name} no ha sido entrenado.")
        
        P_L0 = self.params['P_L0']
        P_T  = self.params['P_T']
        b0 = { 0: self.params['P_IC0'], 1: self.params['P_CC0'] }
        b1 = { 0: self.params['P_IC1'], 1: self.params['P_CC1'] }

        all_predictions, all_actuals = [], []

        for d, sequence in enumerate(test_sequences):
            P_L = P_L0
            for t, (obs_y, c_t) in enumerate(sequence):
                # 1. Probabilidad base BKT
                P_correct_base = P_L * b1[1] + (1 - P_L) * b0[1]

                # 2. Regresión logística con parámetros optimizados
                p_safe = np.clip(P_correct_base, 0.01, 0.99)
                logit_base = np.log(p_safe / (1 - p_safe))
                logit_final = logit_base + (ew * (c_t - np_val))
                P_correct = 1 / (1 + np.exp(-logit_final))

                all_predictions.append(P_correct)
                all_actuals.append(obs_y)

                # 3. Actualización y Transición
                numerator = P_L * b1[obs_y]
                denominator = P_L * b1[obs_y] + (1 - P_L) * b0[obs_y]
                P_L_after_obs = numerator / denominator if denominator > 0 else P_L
                P_L = P_L_after_obs + (1 - P_L_after_obs) * P_T

        return all_predictions, all_actuals
    
    def get_params(self) -> dict:
        return self.params if self.params else {}