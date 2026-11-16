"""Cuenta cuántas veces se repiten las operaciones del algoritmo secuencial.

Evidencia para la tabla de operaciones repetitivas del Paso 1
(informe/01_analisis_secuencial.md).

Usa copias instrumentadas de merge / merge_sort / histograma (la misma lógica
de secuencial/histograma_seq.py, pero con contadores) para no ensuciar la
versión que se mide en los benchmarks.

Uso, desde la raíz del repo:
    python -m benchmarks.contar_operaciones
"""

import math

from comun.config import CUBETAS, TAMANOS
from comun.datos import generar_secuencial
from secuencial.histograma_seq import limites

conteo = {}


def reiniciar():
    for clave in ("llamadas_merge_sort", "llamadas_merge", "comparaciones", "escrituras_aux", "copias_regreso"):
        conteo[clave] = 0


def merge(a, aux, izq, centro, der):
    conteo["llamadas_merge"] += 1
    i, j, k = izq, centro + 1, izq
    while i <= centro and j <= der:
        conteo["comparaciones"] += 1
        if a[i] <= a[j]:
            aux[k] = a[i]
            i += 1
        else:
            aux[k] = a[j]
            j += 1
        k += 1
        conteo["escrituras_aux"] += 1
    while i <= centro:
        aux[k] = a[i]
        i += 1
        k += 1
        conteo["escrituras_aux"] += 1
    while j <= der:
        aux[k] = a[j]
        j += 1
        k += 1
        conteo["escrituras_aux"] += 1
    for p in range(izq, der + 1):
        a[p] = aux[p]
        conteo["copias_regreso"] += 1


def merge_sort(a, aux, izq, der):
    conteo["llamadas_merge_sort"] += 1
    if izq < der:
        centro = (izq + der) // 2
        merge_sort(a, aux, izq, centro)
        merge_sort(a, aux, centro + 1, der)
        merge(a, aux, izq, centro, der)


def contar_histograma(ordenado):
    """Iteraciones del while de histograma() y veces que se cambia de cubeta."""
    n = len(ordenado)
    minimo, maximo, ancho = limites(ordenado)
    iteraciones = cambios = i = cubeta = 0
    lim_inf, lim_sup = minimo, minimo + ancho
    while cubeta < CUBETAS and i < n:
        iteraciones += 1
        x = ordenado[i]
        if cubeta == CUBETAS - 1:
            if lim_inf <= x <= maximo:
                i += 1
            else:
                cubeta += 1
        elif lim_inf <= x < lim_sup:
            i += 1
        else:
            cubeta += 1
            cambios += 1
            lim_inf = minimo + cubeta * ancho
            lim_sup = minimo + (cubeta + 1) * ancho
    return iteraciones, cambios


def main():
    print("| N | Llamadas MergeSort | Llamadas Merge | Comparaciones | Escrituras en aux | "
          "Copias de regreso | Iteraciones histograma | Cambios de cubeta | log₂N |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in TAMANOS:
        reiniciar()
        lista = generar_secuencial(n).tolist()
        merge_sort(lista, [0.0] * n, 0, n - 1)
        iteraciones, cambios = contar_histograma(lista)
        print(f"| {n:,} | {conteo['llamadas_merge_sort']:,} | {conteo['llamadas_merge']:,} | "
              f"{conteo['comparaciones']:,} | {conteo['escrituras_aux']:,} | {conteo['copias_regreso']:,} | "
              f"{iteraciones:,} | {cambios} | {math.log2(n):.2f} |")


if __name__ == "__main__":
    main()
