"""Generación de temperaturas compartida por ambas versiones.

Reproduce la fórmula del C original:
    A[i] = -100.0 + (rand() % 20001) / 100.0;

Los datos se generan por bloques, cada uno con una semilla derivada de
SEMILLA. La versión secuencial recorre los bloques en un loop y la paralela
los reparte en un multiprocessing.Pool; en ambos casos el arreglo final es
exactamente el mismo.
"""

import numpy as np

from comun.config import BLOQUES_DATOS, SEMILLA


def semillas_bloques(n, semilla=SEMILLA, bloques=BLOQUES_DATOS):
    """Divide N en `bloques` partes y devuelve [(semilla_hija, tam), ...]."""
    bloques = max(1, min(bloques, n))
    hijas = np.random.SeedSequence(semilla).spawn(bloques)
    base, resto = divmod(n, bloques)
    tams = [base + (1 if b < resto else 0) for b in range(bloques)]
    return list(zip(hijas, tams))


def generar_bloque(semilla_hija, tam):
    """Genera `tam` temperaturas en [-100, 100] con 2 decimales (float32)."""
    rng = np.random.default_rng(semilla_hija)
    enteros = rng.integers(0, 20001, size=tam)
    return (-100.0 + enteros / 100.0).astype(np.float32)


def generar_secuencial(n, semilla=SEMILLA):
    """Versión de referencia en un solo hilo (línea base de la fase F1)."""
    partes = [generar_bloque(s, t) for s, t in semillas_bloques(n, semilla)]
    return np.concatenate(partes)
