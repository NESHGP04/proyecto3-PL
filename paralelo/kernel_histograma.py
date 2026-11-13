"""Kernel CUDA del histograma (Numba).

Responsable: Camila (rama camila/cpu-histograma)

Diseño (mismo enfoque de "histogramas locales que se combinan" de la EC2):
  1. Cada bloque crea hist_local = cuda.shared.array(CUBETAS, int32) en 0.
  2. Grid-stride loop: cada hilo calcula c = int((x - minimo) / ancho),
     lo acota a CUBETAS-1 y hace cuda.atomic.add(hist_local, c, 1).
  3. cuda.syncthreads()
  4. Los primeros CUBETAS hilos suman hist_local al histograma global con
     cuda.atomic.add(hist_global, c, hist_local[c]).
"""

from numba import cuda  # noqa: F401

from comun.config import CUBETAS, HILOS_POR_BLOQUE  # noqa: F401


# @cuda.jit
# def kernel_histograma(datos, minimo, ancho, hist_global):
#     ...


def histograma_gpu(d_ordenado, minimo, maximo):
    """Recibe el arreglo YA en la GPU y devuelve el DeviceNDArray int32[CUBETAS].

    No copia nada al host: la copia D->H la mide histograma_cuda.py (fase F5).
    """
    raise NotImplementedError("TODO Camila")
