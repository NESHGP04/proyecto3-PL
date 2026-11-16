

import numpy as np
from numba import cuda

from comun.config import HILOS_POR_BLOQUE


@cuda.jit
def kernel_merge_pasada(src, dst, ancho, n, num_merges):
    """Fusiona parejas de segmentos contiguos de tamaño ancho en dst.

    Usa grid-stride loop para procesar las num_merges fusiones de forma segura.
    """
    tid = cuda.grid(1)
    salto = cuda.gridsize(1)

    for t in range(tid, num_merges, salto):
        inicio = t * (2 * ancho)
        medio = inicio + ancho
        if medio > n:
            medio = n
        fin = inicio + 2 * ancho
        if fin > n:
            fin = n

        i = inicio
        j = medio
        k = inicio

        # Fusión ordenada de ambas mitades
        while i < medio and j < fin:
            if src[i] <= src[j]:
                dst[k] = src[i]
                i += 1
            else:
                dst[k] = src[j]
                j += 1
            k += 1

        # Elementos restantes de la primera mitad
        while i < medio:
            dst[k] = src[i]
            i += 1
            k += 1

        # Elementos restantes de la segunda mitad
        while j < fin:
            dst[k] = src[j]
            j += 1
            k += 1


def mergesort_gpu(d_datos):
    """Ordena un DeviceNDArray float32 y devuelve el DeviceNDArray ordenado.

    No hace transferencias host<->device: todo el ordenamiento se realiza
    en memoria de GPU alternando buffers.
    """
    if d_datos.ndim != 1 or d_datos.dtype != np.dtype(np.float32):
        raise ValueError("d_datos debe ser un vector float32 en el dispositivo")

    n = d_datos.size
    if n <= 1:
        return d_datos

    # Buffer auxiliar en GPU para el intercambio alternado (ping-pong)
    d_aux = cuda.device_array_like(d_datos)
    src = d_datos
    dst = d_aux
    ancho = 1

    while ancho < n:
        num_merges = (n + 2 * ancho - 1) // (2 * ancho)
        hilos = HILOS_POR_BLOQUE
        bloques = min((num_merges + hilos - 1) // hilos, 65535)

        kernel_merge_pasada[bloques, hilos](src, dst, ancho, n, num_merges)

        # Alternar buffers para la siguiente pasada
        src, dst = dst, src
        ancho *= 2

    # src contiene el arreglo completamente ordenado en GPU
    return src
