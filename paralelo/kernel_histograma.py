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

import numpy as np
from numba import cuda, float64, int32

from comun.config import CUBETAS, HILOS_POR_BLOQUE


@cuda.jit
def kernel_histograma(datos, minimo, ancho, hist_global):
    """Una tabla compartida de 400 bytes por bloque; dos barreras uniformes."""
    hist_local = cuda.shared.array(CUBETAS, int32)
    tid = cuda.threadIdx.x
    # También funciona si una prueba usa menos de 100 hilos por bloque.
    for c in range(tid, CUBETAS, cuda.blockDim.x):
        hist_local[c] = 0
    cuda.syncthreads()

    for i in range(cuda.grid(1), datos.size, cuda.gridsize(1)):
        c = 0
        if ancho > 0.0:
            x = float64(datos[i])
            c = int((x - minimo) / ancho)
            c = max(0, min(c, CUBETAS - 1))
            # Corregir cocientes casi enteros con los límites del secuencial.
            if c > 0 and x < minimo + c * ancho:
                c -= 1
            if c < CUBETAS - 1 and x >= minimo + (c + 1) * ancho:
                c += 1
        # Si min == max, todos los elementos se asignan a la cubeta cero.
        cuda.atomic.add(hist_local, c, 1)
    cuda.syncthreads()

    for c in range(tid, CUBETAS, cuda.blockDim.x):
        cuda.atomic.add(hist_global, c, hist_local[c])


@cuda.jit
def _inicializar_histograma(hist_global):
    c = cuda.grid(1)
    if c < CUBETAS:
        hist_global[c] = 0


def histograma_gpu(d_ordenado, minimo, maximo):
    """Recibe el arreglo YA en la GPU y devuelve el DeviceNDArray int32[CUBETAS].

    No copia nada al host: la copia D->H la mide histograma_cuda.py (fase F5).
    """
    if d_ordenado.ndim != 1 or d_ordenado.dtype != np.dtype(np.float32):
        raise ValueError("d_ordenado debe ser un vector float32 en el dispositivo")
    if d_ordenado.size > np.iinfo(np.int32).max:
        raise ValueError("N excede la capacidad de los contadores int32")
    minimo, maximo = float(minimo), float(maximo)
    if not np.isfinite(minimo) or not np.isfinite(maximo) or maximo < minimo:
        raise ValueError("minimo y maximo deben ser finitos y estar en orden")

    hist_global = cuda.device_array(CUBETAS, dtype=np.int32)
    _inicializar_histograma[1, HILOS_POR_BLOQUE](hist_global)
    if d_ordenado.size:
        bloques = (d_ordenado.size + HILOS_POR_BLOQUE - 1) // HILOS_POR_BLOQUE
        ancho = (maximo - minimo) / CUBETAS
        kernel_histograma[bloques, HILOS_POR_BLOQUE](
            d_ordenado, minimo, ancho, hist_global
        )
    # Ambos lanzamientos usan el stream por defecto y quedan ordenados.
    # No hay to_device, copy_to_host ni synchronize: F5 pertenece al pipeline.
    return hist_global
