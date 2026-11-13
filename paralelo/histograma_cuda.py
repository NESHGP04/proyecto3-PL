"""Pipeline PARALELO completo: CPU (preparación/archivos) + GPU (sort + histograma).

Responsable: Carlos (rama carlos/mergesort-gpu)

    F1 preparación   -> paralelo.preparacion_cpu.preparar_datos      (CPU, Pool)
    F2 H->D          -> cuda.to_device                                (PCIe)
    F3 ordenamiento  -> paralelo.kernel_mergesort.mergesort_gpu       (GPU)
    F4 histograma    -> paralelo.kernel_histograma.histograma_gpu     (GPU)
    F5 D->H          -> copy_to_host                                  (PCIe)
    F6 escritura     -> paralelo.preparacion_cpu.escribir_resultados  (CPU, Pool)

Medir los kernels con cuda.event() (+ cuda.synchronize()) y las fases de CPU
con time.perf_counter(). min y max salen de d_ordenado[0] y d_ordenado[n-1].
"""


def ejecutar(n, escribir_archivos=True, carpeta_salida="resultados"):
    """Devuelve (ordenado, hist, tiempos) con las mismas llaves que el secuencial."""
    raise NotImplementedError("TODO Carlos")


def validar(n=100_000):
    """Compara contra np.sort y contra secuencial.histograma_seq.histograma."""
    raise NotImplementedError("TODO Carlos")
