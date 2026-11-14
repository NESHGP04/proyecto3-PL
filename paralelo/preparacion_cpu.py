"""Fase CPU del pipeline paralelo: preparación de datos y escritura de archivos.

Responsable: Camila (rama camila/cpu-histograma)

Aquí NO va matemática pesada: solo generar o cargar datos y manipular archivos,
repartido entre los núcleos del CPU con multiprocessing.
"""

import operator
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from comun.config import BLOQUES_DATOS, CUBETAS, TEMP_MIN, TEMP_MAX
from comun.datos import generar_bloque, semillas_bloques


def _procesos(procesos, tareas):
    """No crear más procesos que bloques de trabajo."""
    cantidad = (os.cpu_count() or 1) if procesos is None else operator.index(procesos)
    if cantidad < 1:
        raise ValueError("procesos debe ser un entero positivo")
    return min(cantidad, max(1, tareas))


def preparar_datos(n, procesos=None):
    """F1: genera las N temperaturas en paralelo y devuelve np.ndarray float32.

    Usar Pool.starmap(generar_bloque, semillas_bloques(n)) y concatenar
    EN ORDEN, para que el resultado sea idéntico a comun.datos.generar_secuencial(n).
    """
    n = operator.index(n)
    if n < 0:
        raise ValueError("n debe ser un entero no negativo")
    tareas = semillas_bloques(n)
    trabajadores = _procesos(procesos, len(tareas))
    if n == 0:
        return np.empty(0, dtype=np.float32)
    with Pool(processes=trabajadores) as pool:
        partes = pool.starmap(generar_bloque, tareas)
    return np.concatenate(partes)


def _formatear_bloque(tarea):
    """Función de módulo para que multiprocessing pueda usar spawn (Mac/Colab)."""
    inicio, valores = tarea
    return "".join(f"{inicio + i},{float(x):.2f}\n" for i, x in enumerate(valores))


def escribir_resultados(ordenado, hist, carpeta_salida="resultados", procesos=None):
    """F6: escribe temperaturas_ordenadas.csv, histograma.dat e histograma.png.

    Formatear el CSV por bloques en paralelo (cada worker devuelve un string)
    y escribirlos en orden en el archivo.
    """
    ordenado = np.asarray(ordenado)
    hist = np.asarray(hist)
    if ordenado.ndim != 1 or not np.all(np.isfinite(ordenado)):
        raise ValueError("ordenado debe ser un vector de temperaturas finitas")
    if hist.shape != (CUBETAS,) or not np.issubdtype(hist.dtype, np.integer):
        raise ValueError(f"hist debe contener {CUBETAS} contadores enteros")
    if np.any(hist < 0) or int(hist.sum(dtype=np.int64)) != ordenado.size:
        raise ValueError("hist debe ser no negativo y su suma debe ser N")

    n = ordenado.size
    tam = max(1, min(65_536, (n + BLOQUES_DATOS - 1) // BLOQUES_DATOS))
    tareas = ((i, ordenado[i:i + tam]) for i in range(0, n, tam))
    trabajadores = _procesos(procesos, (n + tam - 1) // tam)
    salida = Path(carpeta_salida)
    salida.mkdir(parents=True, exist_ok=True)
    rutas = {nombre: salida / nombre for nombre in (
        "temperaturas_ordenadas.csv", "histograma.dat", "histograma.png"
    )}

    # imap conserva el orden; solo el padre abre y escribe el archivo compartido.
    with rutas["temperaturas_ordenadas.csv"].open("w", encoding="utf-8", newline="") as archivo:
        archivo.write("Indice,Temperatura\n")
        if n:
            with Pool(processes=trabajadores) as pool:
                for bloque in pool.imap(_formatear_bloque, tareas, chunksize=1):
                    archivo.write(bloque)

    minimo = float(ordenado[0]) if n else TEMP_MIN
    maximo = float(ordenado[-1]) if n else TEMP_MAX
    ancho = (maximo - minimo) / CUBETAS
    limites = minimo + np.arange(CUBETAS, dtype=np.float64) * ancho
    # El C escribe el límite inferior, aunque su comentario dice "valor central".
    with rutas["histograma.dat"].open("w", encoding="utf-8") as archivo:
        for limite, frecuencia in zip(limites, hist):
            archivo.write(f"{limite:.2f} {int(frecuencia)}\n")

    # Figure + Agg funciona en Colab y terminal, sin ventanas ni backend interactivo.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figura = Figure(figsize=(14, 7), dpi=100)
    FigureCanvasAgg(figura)
    eje = figura.subplots()
    if ancho > 0:
        eje.bar(limites, hist, width=ancho, align="edge", color="#159878")
        eje.set_xlim(minimo, maximo)
    else:
        eje.bar([minimo], [int(hist.sum())], width=1.0, color="#159878")
        eje.set_xlim(minimo - 1, minimo + 1)
    eje.set(title="Histograma de temperaturas", xlabel="Temperatura (°C)", ylabel="Frecuencia")
    eje.grid(axis="y", alpha=0.25)
    figura.tight_layout()
    figura.savefig(rutas["histograma.png"])
    figura.clear()
    return rutas
