"""Versión SECUENCIAL: traducción fiel de original_c/histograma.c a Python.

Responsable: Marinés (rama marines/secuencial)

Es la línea base para calcular el speedup. Todo corre en un solo núcleo:
    F1 preparación   -> comun.datos.generar_secuencial
    F3 ordenamiento  -> merge_sort (recursivo, igual que el C)
    F4 histograma    -> histograma (recorrido por cubetas sobre el arreglo ordenado)
    F6 escritura     -> CSV + histograma.dat + histograma.png
"""

import numpy as np

from comun.config import CUBETAS
from comun.datos import generar_secuencial  # noqa: F401  (se usa en ejecutar)


def merge(arreglo, aux, izq, centro, der):
    """Fusiona arreglo[izq..centro] y arreglo[centro+1..der] usando aux."""
    raise NotImplementedError("TODO Marinés")


def merge_sort(arreglo, aux, izq, der):
    """MergeSort recursivo in-place sobre arreglo[izq..der]."""
    raise NotImplementedError("TODO Marinés")


def histograma(ordenado):
    """Devuelve np.ndarray int32[CUBETAS] recorriendo el arreglo ordenado.

    Regla acordada (PLAN.md §4): límite de la cubeta c = min + c*ancho
    (no acumular limSup += ancho); la última cubeta incluye el máximo.
    """
    raise NotImplementedError("TODO Marinés")


def ejecutar(n, escribir_archivos=True, carpeta_salida="resultados"):
    """Corre el pipeline completo y devuelve (ordenado, hist, tiempos).

    `tiempos` es un dict con las llaves de comun.config.FASES
    (h2d y d2h valen 0.0 en esta versión). Medir con time.perf_counter().
    """
    raise NotImplementedError("TODO Marinés")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    _, hist, tiempos = ejecutar(n)
    assert hist.sum() == n and len(hist) == CUBETAS
    print(tiempos)
