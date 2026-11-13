"""Kernel CUDA del MergeSort bottom-up (Numba).

Responsable: Carlos (rama carlos/mergesort-gpu)

Diseño:
  - La recursión del C se reemplaza por pasadas iterativas con ancho
    w = 1, 2, 4, ..., hasta >= N.
  - En cada pasada se lanza 1 hilo por pareja de subarreglos:
        hilos = ceil(N / (2*w)); bloques = ceil(hilos / HILOS_POR_BLOQUE)
    El hilo t fusiona src[i:i+w] con src[i+w:i+2w] en dst, donde i = 2*w*t.
  - Se alternan los buffers src <-> dst entre pasadas; todo queda en la GPU.
  - Limitación que hay que discutir en el informe: en las últimas pasadas hay
    muy pocos hilos activos (en la última, 1 solo hilo fusiona N elementos).
"""

from numba import cuda  # noqa: F401

from comun.config import HILOS_POR_BLOQUE  # noqa: F401


# @cuda.jit
# def kernel_merge_pasada(src, dst, ancho, n):
#     ...


def mergesort_gpu(d_datos):
    """Ordena un DeviceNDArray float32 y devuelve el DeviceNDArray ordenado.

    No hace transferencias host<->device: eso lo mide histograma_cuda.py.
    """
    raise NotImplementedError("TODO Carlos")
