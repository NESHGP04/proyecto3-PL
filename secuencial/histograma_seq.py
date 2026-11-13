"""Versión SECUENCIAL

Es la línea base para calcular el speedup. Todo corre en un solo núcleo:
    F1 preparación   -> comun.datos.generar_secuencial
    F3 ordenamiento  -> merge_sort (recursivo, igual que el C)
    F4 histograma    -> histograma (recorrido por cubetas sobre el arreglo ordenado)
    F6 escritura     -> CSV + histograma.dat + histograma.png

"""

import os
import time

import numpy as np

from comun.config import CUBETAS, FASES
from comun.datos import generar_secuencial

CARPETA_SALIDA = os.path.join("resultados", "salida_secuencial")


def merge(arreglo, aux, izq, centro, der):
    """Fusiona arreglo[izq..centro] y arreglo[centro+1..der] usando aux.

    Igual que Merge() del C: va tomando el menor de las dos mitades,
    copia lo que sobre de cada mitad y regresa el resultado a arreglo.
    """
    i = izq  # recorre la mitad izquierda
    j = centro + 1  # recorre la mitad derecha
    k = izq  # posición de escritura en aux

    while i <= centro and j <= der:
        if arreglo[i] <= arreglo[j]:  # "<=" mantiene el orden estable
            aux[k] = arreglo[i]
            i += 1
        else:
            aux[k] = arreglo[j]
            j += 1
        k += 1

    # Sobrantes de la mitad izquierda
    while i <= centro:
        aux[k] = arreglo[i]
        i += 1
        k += 1

    # Sobrantes de la mitad derecha
    while j <= der:
        aux[k] = arreglo[j]
        j += 1
        k += 1

    # Copiar de regreso al arreglo original (mismo ciclo que el C)
    for p in range(izq, der + 1):
        arreglo[p] = aux[p]


def merge_sort(arreglo, aux, izq, der):
    """MergeSort recursivo in-place sobre arreglo[izq..der].

    Profundidad de recursión ~ log2(N) (= 20 para N = 10^6), así que no
    hay riesgo de pasar el límite de recursión de Python.
    """
    if izq < der:
        centro = (izq + der) // 2
        merge_sort(arreglo, aux, izq, centro)
        merge_sort(arreglo, aux, centro + 1, der)
        merge(arreglo, aux, izq, centro, der)


def ordenar(datos):
    """Envuelve merge_sort: recibe un np.ndarray y devuelve uno nuevo ordenado.

    Igual que el C, no modifica el arreglo original (A[] -> copia en temp[]).
    """
    lista = datos.tolist()  # copia, como temp[i] = A[i]
    n = len(lista)
    aux = [0.0] * n  # equivalente a tempAux[]
    merge_sort(lista, aux, 0, n - 1)
    return np.array(lista, dtype=np.float32)


def limites(ordenado):
    """Devuelve (minimo, maximo, ancho) del arreglo ordenado, en double.

    La GPU debe usar exactamente estos mismos valores para que los
    histogramas coincidan.
    """
    minimo = float(ordenado[0])
    maximo = float(ordenado[-1])
    ancho = (maximo - minimo) / CUBETAS
    if ancho == 0:  # todos los valores iguales (igual que en el C)
        ancho = 1.0
    return minimo, maximo, ancho


def histograma(ordenado):
    """Devuelve np.ndarray int32[CUBETAS] recorriendo el arreglo ordenado.

    Misma idea que Histograma() del C: como el arreglo está ordenado,
    se avanza una sola vez por los datos y se pasa a la siguiente cubeta
    cuando el valor ya no cae en el rango de la actual.

    Regla acordada (PLAN.md §4):
      - cubeta c cubre [minimo + c*ancho, minimo + (c+1)*ancho)
      - los límites se calculan como minimo + c*ancho en lugar de acumular
        limSup += ancho (acumular arrastra error de redondeo cubeta tras
        cubeta y haría que la GPU no coincida)
      - la última cubeta va hasta el máximo inclusive, para no perderlo
    """
    datos = ordenado.tolist() if isinstance(ordenado, np.ndarray) else ordenado
    n = len(datos)
    hist = [0] * CUBETAS
    if n == 0:
        return np.zeros(CUBETAS, dtype=np.int32)

    minimo, maximo, ancho = limites(datos)

    i = 0
    cubeta = 0
    lim_inf = minimo
    lim_sup = minimo + ancho

    while cubeta < CUBETAS and i < n:
        x = datos[i]
        if cubeta == CUBETAS - 1:
            # Última cubeta: todo lo que queda (<= maximo) entra aquí
            if lim_inf <= x <= maximo:
                hist[cubeta] += 1
                i += 1
            else:
                cubeta += 1
        elif lim_inf <= x < lim_sup:
            hist[cubeta] += 1
            i += 1
        else:
            # Ya no pertenece: pasar a la siguiente cubeta
            cubeta += 1
            lim_inf = minimo + cubeta * ancho
            lim_sup = minimo + (cubeta + 1) * ancho

    return np.array(hist, dtype=np.int32)


def escribir_csv(ordenado, ruta):
    """temperaturas_ordenadas.csv, mismo formato que el C: "Indice,Temperatura"."""
    with open(ruta, "w") as f:
        f.write("Indice,Temperatura\n")
        for i, x in enumerate(ordenado.tolist()):
            f.write(f"{i},{x:.2f}\n")


def escribir_dat(ordenado, hist, ruta):
    """histograma.dat, mismo formato que el C: "<inicio de cubeta> <frecuencia>"."""
    minimo, _, ancho = limites(ordenado)
    with open(ruta, "w") as f:
        for c in range(CUBETAS):
            f.write(f"{minimo + c * ancho:.2f} {int(hist[c])}\n")


def graficar(ordenado, hist, ruta):
    """histograma.png. Reemplaza la llamada a gnuplot del C con matplotlib."""
    import matplotlib

    matplotlib.use("Agg")  # sin ventana: funciona en Colab y en terminal
    import matplotlib.pyplot as plt

    minimo, _, ancho = limites(ordenado)
    inicios = [minimo + c * ancho for c in range(CUBETAS)]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(inicios, hist, width=ancho * 0.9, align="edge", color="#2E86DE", label="Frecuencia")
    ax.set_title("Histograma de Temperaturas")
    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("Frecuencia")
    ax.set_xlim(-100, 100)
    ax.set_xticks(range(-100, 101, 20))
    ax.grid(axis="y", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=100)
    plt.close(fig)


def escribir_resultados(ordenado, hist, carpeta_salida):
    """F6: escribe los 3 archivos de salida en un solo hilo."""
    os.makedirs(carpeta_salida, exist_ok=True)
    escribir_csv(ordenado, os.path.join(carpeta_salida, "temperaturas_ordenadas.csv"))
    escribir_dat(ordenado, hist, os.path.join(carpeta_salida, "histograma.dat"))
    graficar(ordenado, hist, os.path.join(carpeta_salida, "histograma.png"))


def ejecutar(n, escribir_archivos=True, carpeta_salida=CARPETA_SALIDA):
    """Corre el pipeline completo y devuelve (ordenado, hist, tiempos).

    `tiempos` es un dict con las llaves de comun.config.FASES, en segundos.
    h2d y d2h valen 0.0 porque esta versión no usa GPU; se dejan para que
    la tabla de tiempos tenga las mismas columnas que la versión CUDA.
    """
    tiempos = dict.fromkeys(FASES, 0.0)
    inicio = time.perf_counter()

    # F1: preparación de datos
    t = time.perf_counter()
    datos = generar_secuencial(n)
    tiempos["preparacion"] = time.perf_counter() - t

    # F3: ordenamiento (MergeSort recursivo)
    t = time.perf_counter()
    ordenado = ordenar(datos)
    tiempos["ordenamiento"] = time.perf_counter() - t

    # F4: histograma (recorrido de cubetas)
    t = time.perf_counter()
    hist = histograma(ordenado)
    tiempos["histograma"] = time.perf_counter() - t

    # F6: escritura de archivos
    if escribir_archivos:
        t = time.perf_counter()
        escribir_resultados(ordenado, hist, carpeta_salida)
        tiempos["escritura"] = time.perf_counter() - t

    tiempos["total"] = time.perf_counter() - inicio
    return ordenado, hist, tiempos


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    _, hist, tiempos = ejecutar(n)
    assert hist.sum() == n and len(hist) == CUBETAS

    print(f"Versión secuencial, N = {n:,}")
    for fase in FASES:
        print(f"  {fase:<13} {tiempos[fase]:9.4f} s")
    print(f"Archivos en {CARPETA_SALIDA}/")
