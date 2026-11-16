

import time
import numpy as np
from numba import cuda

from comun.config import CUBETAS, FASES, TEMP_MAX, TEMP_MIN
from comun.datos import generar_secuencial
from paralelo.kernel_histograma import histograma_gpu
from paralelo.kernel_mergesort import mergesort_gpu
from paralelo.preparacion_cpu import escribir_resultados, preparar_datos


def ejecutar(n, escribir_archivos=True, carpeta_salida="resultados", procesos=None):
    """Devuelve (ordenado, hist, tiempos) con las mismas llaves que el secuencial.

    Mide fases de GPU con cuda.event() y fases de CPU con time.perf_counter().
    """
    tiempos = dict.fromkeys(FASES, 0.0)
    inicio_total = time.perf_counter()

    # Eventos CUDA para medir GPU y transferencias
    ev_inicio = cuda.event()
    ev_fin = cuda.event()

    # F1: Preparación de datos (CPU multinúcleo con Pool)
    t = time.perf_counter()
    datos = preparar_datos(n, procesos=procesos)
    tiempos["preparacion"] = time.perf_counter() - t

    # F2: Transferencia Host -> Device
    ev_inicio.record()
    d_datos = cuda.to_device(datos)
    ev_fin.record()
    ev_fin.synchronize()
    tiempos["h2d"] = cuda.event_elapsed_time(ev_inicio, ev_fin) / 1000.0

    # F3: Ordenamiento MergeSort en GPU
    ev_inicio.record()
    d_ordenado = mergesort_gpu(d_datos)
    ev_fin.record()
    ev_fin.synchronize()
    tiempos["ordenamiento"] = cuda.event_elapsed_time(ev_inicio, ev_fin) / 1000.0

    # Obtener extremos para los límites del histograma (1 float32 cada uno)
    minimo = float(d_ordenado[:1].copy_to_host()[0]) if n > 0 else TEMP_MIN
    maximo = float(d_ordenado[-1:].copy_to_host()[0]) if n > 0 else TEMP_MAX

    # F4: Histograma en GPU (memoria compartida + atómicos)
    ev_inicio.record()
    d_hist = histograma_gpu(d_ordenado, minimo, maximo)
    ev_fin.record()
    ev_fin.synchronize()
    tiempos["histograma"] = cuda.event_elapsed_time(ev_inicio, ev_fin) / 1000.0

    # F5: Transferencia Device -> Host
    ev_inicio.record()
    ordenado = d_ordenado.copy_to_host()
    hist = d_hist.copy_to_host()
    ev_fin.record()
    ev_fin.synchronize()
    tiempos["d2h"] = cuda.event_elapsed_time(ev_inicio, ev_fin) / 1000.0

    # F6: Escritura de archivos (CPU, formateo paralelo y escritura en orden)
    if escribir_archivos:
        t = time.perf_counter()
        escribir_resultados(ordenado, hist, carpeta_salida=carpeta_salida, procesos=procesos)
        tiempos["escritura"] = time.perf_counter() - t

    tiempos["total"] = time.perf_counter() - inicio_total
    return ordenado, hist, tiempos


def validar(n=100_000):
    """Compara contra np.sort y contra secuencial.histograma_seq.histograma."""
    from secuencial.histograma_seq import histograma as histograma_seq

    print(f"Iniciando validación con N = {n:,}...")
    datos_ref = generar_secuencial(n)
    esperado_ordenado = np.sort(datos_ref)
    esperado_hist = histograma_seq(esperado_ordenado)

    ordenado, hist, _ = ejecutar(n, escribir_archivos=False)

    # Validaciones obligatorias (PLAN.md §4)
    np.testing.assert_allclose(
        ordenado,
        esperado_ordenado,
        rtol=1e-5,
        atol=1e-5,
        err_msg="El ordenamiento GPU no coincide exactamente con np.sort"
    )
    np.testing.assert_array_equal(
        hist,
        esperado_hist,
        err_msg="El histograma GPU no coincide con el histograma secuencial"
    )
    assert int(hist.sum()) == n, f"Suma de frecuencias ({hist.sum()}) != N ({n})"
    assert len(hist) == CUBETAS, f"Número de cubetas ({len(hist)}) != {CUBETAS}"

    print(f" Validación exitosa para N = {n:,}:")
    print(f"   - Ordenamiento GPU coincide con np.sort")
    print(f"   - Histograma GPU coincide con secuencial ({CUBETAS} cubetas)")
    print(f"   - Suma total de frecuencias = {int(hist.sum()):,}")
    return True


if __name__ == "__main__":
    import sys
    n_prueba = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    validar(n_prueba)
