import numpy as np
from typing import List, Tuple, Dict

class BaumWelchREBKT:
    """
    Implementación del algoritmo Baum-Welch para el modelo RE-BKT.
    Matriz de observación B de dimensión 2×4:
        Filas  → estados ocultos: 0=No Dominado, 1=Dominado
        Columnas → observaciones compuestas:
            0: (Incorrecto, Concentrado)    → P_IC0 / P_IC1
            1: (Incorrecto, No Concentrado) → P_INC0 / P_INC1
            2: (Correcto,   Concentrado)    → P_CC0  / P_CC1
            3: (Correcto,   No Concentrado) → P_CNC0 / P_CNC1
    """

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-4):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.threshold = 0.5
        #print(f'Umbral en baumwelch {threshold}')

        # Parámetros del modelo (se inicializan en fit)
        self.P_L0 = None  # Probabilidad inicial de conocer la habilidad
        self.P_T  = None  # Probabilidad de aprendizaje (transición)

        # Emisión estado No Dominado (fila 0)
        self.P_IC0  = None  # Incorrecto + Concentrado
        self.P_INC0 = None  # Incorrecto + No Concentrado
        self.P_CC0  = None  # Correcto   + Concentrado
        self.P_CNC0 = None  # Correcto   + No Concentrado

        # Emisión estado Dominado (fila 1)
        self.P_IC1  = None
        self.P_INC1 = None
        self.P_CC1  = None
        self.P_CNC1 = None

    def initialize_parameters(self, seed: int = 42):
        """Inicializa los parámetros con valores aleatorios controlados."""
        np.random.seed(seed)
        self.P_L0 = np.random.uniform(0.1, 0.3)
        self.P_T  = np.random.uniform(0.1, 0.3)

        # Inicialización de matriz de observación (B)
        # Fila 0 — No Dominado: 4 valores normalizados
        row0 = np.random.uniform(0.01, 1.0, 4)
        row0 /= row0.sum()
        self.P_IC0, self.P_INC0, self.P_CC0, self.P_CNC0 = row0

        # Fila 1 — Dominado: 4 valores normalizados
        row1 = np.random.uniform(0.01, 1.0, 4)
        row1 /= row1.sum()
        self.P_IC1, self.P_INC1, self.P_CC1, self.P_CNC1 = row1

    def get_params(self) -> Dict[str, float]:
        """Retorna un diccionario con los parámetros actuales del modelo."""
        return {
            'P_L0': self.P_L0, 'P_T': self.P_T,
            # No Dominado
            'P_IC0': self.P_IC0, 'P_INC0': self.P_INC0,
            'P_CC0': self.P_CC0, 'P_CNC0': self.P_CNC0,
            # Dominado
            'P_IC1': self.P_IC1, 'P_INC1': self.P_INC1,
            'P_CC1': self.P_CC1, 'P_CNC1': self.P_CNC1,
        }

    def get_initial_probabilities(self) -> np.ndarray:
        return np.array([1 - self.P_L0, self.P_L0])

    def get_transition_matrix(self) -> np.ndarray:
        return np.array([
            [1 - self.P_T, self.P_T],
            [0.0,          1.0     ]
        ])

    def get_observation_matrix(self) -> np.ndarray:
        """Matriz B dimensión 2x4"""
        return np.array([
            [self.P_IC0, self.P_INC0, self.P_CC0, self.P_CNC0],  # No Dominado
            [self.P_IC1, self.P_INC1, self.P_CC1, self.P_CNC1],  # Dominado
        ])

    def get_params(self) -> dict:
        return {
            'P_L0': self.P_L0, 'P_T': self.P_T,
            'P_IC0': self.P_IC0, 'P_INC0': self.P_INC0, 'P_CC0': self.P_CC0, 'P_CNC0': self.P_CNC0,
            'P_IC1': self.P_IC1, 'P_INC1': self.P_INC1, 'P_CC1': self.P_CC1, 'P_CNC1': self.P_CNC1,
        }

    # PASO E

    def forward_algorithm(self, y: List[Tuple[int, float]]) -> Tuple[np.ndarray, float]:
        T  = len(y)
        pi = self.get_initial_probabilities()
        A  = self.get_transition_matrix()
        B  = self.get_observation_matrix()

        alpha = np.zeros((T, 2))

        y_0, c_0 = y[0]
        alpha[0, 0] = pi[0] * B[0, y_0]
        alpha[0, 1] = pi[1] * B[1, y_0]

        for t in range(T - 1):
            y_next, c_next = y[t+1]
            alpha[t+1, 0] = (alpha[t, 0] * A[0, 0] + alpha[t, 1] * A[1, 0]) * B[0, y_next]
            alpha[t+1, 1] = (alpha[t, 0] * A[0, 1] + alpha[t, 1] * A[1, 1]) * B[1, y_next]

        log_likelihood = np.log(alpha[-1, 0] + alpha[-1, 1] + 1e-10)
        return alpha, log_likelihood

    def backward_algorithm(self, y: List[Tuple[int, float]]) -> np.ndarray:
        T = len(y)
        A = self.get_transition_matrix()
        B = self.get_observation_matrix()

        beta = np.zeros((T, 2))
        beta[T-1, 0] = 1.0
        beta[T-1, 1] = 1.0

        for t in range(T - 2, -1, -1):
            y_next, c_next = y[t+1]
            beta[t, 0] = (beta[t+1, 0] * A[0, 0] * B[0, y_next] +
                          beta[t+1, 1] * A[0, 1] * B[1, y_next])
            beta[t, 1] = (beta[t+1, 0] * A[1, 0] * B[0, y_next] +
                          beta[t+1, 1] * A[1, 1] * B[1, y_next])
        return beta

    def compute_gamma(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        numerator   = alpha * beta
        denominator = numerator.sum(axis=1, keepdims=True)
        return numerator / (denominator + 1e-10)

    def compute_xi(self, y: List[Tuple[int, float]], alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        T = len(y)
        A = self.get_transition_matrix()
        B = self.get_observation_matrix()
        xi = np.zeros((T - 1, 2, 2))

        for t in range(T - 1):
            y_next, c_next = y[t+1]
            denom = 0.0
            for k in range(2):
                for w in range(2):
                    denom += alpha[t, k] * A[k, w] * B[w, y_next] * beta[t+1, w]

            for i in range(2):
                for j in range(2):
                    num = alpha[t, i] * A[i, j] * B[j, y_next] * beta[t+1, j]
                    xi[t, i, j] = num / (denom + 1e-10)
        return xi

    # PASO M

    def m_step(self, observations_list: List[List[Tuple[int, float]]],
               gamma_list: List[np.ndarray], xi_list: List[np.ndarray], threshold=None) -> Dict[str, float]:
        #threshold = getattr(self, 'threshold', 0.5)
        #threshold = threshold if threshold is not None else self.threshold
        #print(f'Umbral en baumwelch {threshold}')
        D = len(observations_list)

        sum_gamma_1_start = 0.0
        numerator_T   = 0.0
        denominator_T = 0.0

        denominator_B0 = 0.0
        denominator_B1 = 0.0

        # Numeradores para cada observación — No Dominado
        numerator_IC0  = 0.0
        numerator_INC0 = 0.0
        numerator_CC0  = 0.0
        numerator_CNC0 = 0.0

        # Numeradores para cada observación — Dominado
        numerator_IC1  = 0.0
        numerator_INC1 = 0.0
        numerator_CC1  = 0.0
        numerator_CNC1 = 0.0


        for d in range(D):
            observations = observations_list[d]
            gamma        = gamma_list[d]
            xi           = xi_list[d]
            T_d          = len(observations)

            sum_gamma_1_start += gamma[0, 1]

            for t in range(T_d):
                y_t, c_t = observations[t]  # c_t es el valor continuo [0, 1]

                denominator_B0 += gamma[t, 0]
                denominator_B1 += gamma[t, 1]

                # Definir si está concentrado
                is_concentrated = (c_t >= self.threshold)

                if is_concentrated:
                    if y_t == 1:
                        numerator_CC0 += gamma[t, 0]
                        numerator_CC1 += gamma[t, 1]

                    else:
                        numerator_IC0 += gamma[t, 0]
                        numerator_IC1 += gamma[t, 1]
                else:
                    if y_t == 1:
                        numerator_CNC0 += gamma[t, 0]
                        numerator_CNC1 += gamma[t, 1]

                    else:
                        numerator_INC0 += gamma[t, 0]
                        numerator_INC1 += gamma[t, 1]

            # Transiciones (Aprender)
            for t in range(T_d - 1):
                numerator_T   += xi[t, 0, 1]
                denominator_T += gamma[t, 0]

        # Calculo de nuevos parámetros
        new_P_L0 = sum_gamma_1_start / D
        new_P_T = numerator_T / denominator_T if denominator_T > 0 else self.P_T

        # Estado 0 (No Dominado)
        new_P_IC0 = numerator_IC0 / denominator_B0 if denominator_B0 > 0 else self.P_IC0
        new_P_CC0 = numerator_CC0 / denominator_B0 if denominator_B0 > 0 else self.P_CC0
        new_P_INC0 = numerator_INC0 / denominator_B0 if denominator_B0 > 0 else self.P_INC0
        new_P_CNC0 = numerator_CNC0 / denominator_B0 if denominator_B0 > 0 else self.P_CNC0

        # Estado 1 (Dominado)
        new_P_IC1 = numerator_IC1 / denominator_B1 if denominator_B1 > 0 else self.P_IC1
        new_P_CC1 = numerator_CC1 / denominator_B1 if denominator_B1 > 0 else self.P_CC1
        new_P_INC1 = numerator_INC1 / denominator_B1 if denominator_B1 > 0 else self.P_INC1
        new_P_CNC1 = numerator_CNC1 / denominator_B1 if denominator_B1 > 0 else self.P_CNC1

        #Normalizados
        new_P_L0 = np.clip(new_P_L0, 0.01, 0.99)
        new_P_T  = np.clip(new_P_T,  0.01, 0.99)

        new_P_IC0  = np.clip(new_P_IC0, 0.01, 0.99)
        new_P_INC0 = np.clip(new_P_INC0, 0.01, 0.99)
        new_P_CC0  = np.clip(new_P_CC0, 0.01, 0.99)
        new_P_CNC0 = np.clip(new_P_CNC0, 0.01, 0.99)

        row0 = np.array([new_P_IC0, new_P_INC0, new_P_CC0, new_P_CNC0])
        row0 /= row0.sum() # Garantiza que las 4 probabilidades sumen 1.0

        new_P_IC1  = np.clip(new_P_IC1, 0.01, 0.99)
        new_P_INC1 = np.clip(new_P_INC1, 0.01, 0.99)
        new_P_CC1  = np.clip(new_P_CC1, 0.01, 0.99)
        new_P_CNC1 = np.clip(new_P_CNC1, 0.01, 0.99)

        row1 = np.array([new_P_IC1, new_P_INC1, new_P_CC1, new_P_CNC1])
        row1 /= row1.sum() # Garantiza que las 4 probabilidades sumen 1.0

        return {
            'P_L0': new_P_L0, 'P_T': new_P_T,
            'P_IC0': row0[0], 'P_INC0': row0[1], 'P_CC0': row0[2], 'P_CNC0': row0[3],
            'P_IC1': row1[0], 'P_INC1': row1[1], 'P_CC1': row1[2], 'P_CNC1': row1[3]
        }

    def fit(self, observations_list: List[List[Tuple[int, float]]], threshold=None, verbose: bool = False):
        self.initialize_parameters()
        self.history = []
        self.threshold = threshold if threshold is not None else self.threshold
        prev_ll = -np.inf

        print("=" * 60)
        print("BAUM-WELCH  —  Modelo RE-BKT (Concentración Continua)")
        print("=" * 60)

        for iteration in range(self.max_iterations):
            gamma_list, xi_list = [], []
            total_ll   = 0.0

            for observations in observations_list:
                alpha, log_likelihood = self.forward_algorithm(observations)
                beta       = self.backward_algorithm(observations)
                gamma      = self.compute_gamma(alpha, beta)
                xi         = self.compute_xi(observations, alpha, beta)

                gamma_list.append(gamma)
                xi_list.append(xi)
                total_ll += log_likelihood

            self.history.append(total_ll)
            progress = total_ll - prev_ll
            #print(f"Iter {iteration+1:3d} | log-lik = {total_ll:.6f} | diferencia = {progress:.2e}")

            if abs(progress) < self.tolerance:
                print(f"\n¡Convergencia en iteración {iteration + 1}!")
                break

            new_params = self.m_step(observations_list, gamma_list, xi_list, threshold)
            for key, val in new_params.items():
                setattr(self, key, val)

            prev_ll = total_ll

        print("\nParámetros finales:")
        for k, v in self.get_params().items():
            print(f"  {k} = {v:.6f}")