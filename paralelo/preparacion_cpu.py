"""Fase CPU del pipeline paralelo: preparación de datos y escritura de archivos.

Responsable: Camila (rama camila/cpu-histograma)

Aquí NO va matemática pesada: solo generar o cargar datos y manipular archivos,
repartido entre los núcleos del CPU con multiprocessing.
"""

from multiprocessing import Pool  # noqa: F401

from comun.datos import generar_bloque, semillas_bloques  # noqa: F401


def preparar_datos(n, procesos=None):
    """F1: genera las N temperaturas en paralelo y devuelve np.ndarray float32.

    Usar Pool.starmap(generar_bloque, semillas_bloques(n)) y concatenar
    EN ORDEN, para que el resultado sea idéntico a comun.datos.generar_secuencial(n).
    """
    raise NotImplementedError("TODO Camila")


def escribir_resultados(ordenado, hist, carpeta_salida="resultados", procesos=None):
    """F6: escribe temperaturas_ordenadas.csv, histograma.dat e histograma.png.

    Formatear el CSV por bloques en paralelo (cada worker devuelve un string)
    y escribirlos en orden en el archivo.
    """
    raise NotImplementedError("TODO Camila")
