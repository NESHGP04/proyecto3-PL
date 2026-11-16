"""Pruebas de la versión secuencial (no necesitan GPU).

Uso, desde la raíz del repo:
    python -m tests.test_secuencial
"""

import os
import tempfile
import time

import numpy as np

from comun.config import CUBETAS, FASES
from comun.datos import generar_secuencial
from secuencial.histograma_seq import ejecutar, histograma, limites, ordenar


def referencia_histograma(ordenado):
    """Histograma con la fórmula directa que usará la GPU: c = int((x - min) / ancho).

    No recorre cubetas: calcula el índice de cada elemento. Sirve para
    comprobar que el recorrido secuencial y el kernel cuentan igual.
    """
    minimo, _, ancho = limites(ordenado)
    x = ordenado.astype(np.float64)
    idx = ((x - minimo) / ancho).astype(np.int64)
    idx = np.clip(idx, 0, CUBETAS - 1)
    return np.bincount(idx, minlength=CUBETAS).astype(np.int32)


def prueba_ordenamiento():
    casos = {
        "un elemento": np.array([5.0], dtype=np.float32),
        "dos al revés": np.array([3.0, -1.0], dtype=np.float32),
        "ya ordenado": np.arange(50, dtype=np.float32),
        "al revés": np.arange(50, dtype=np.float32)[::-1].copy(),
        "repetidos": np.array([2, 1, 2, 1, 2, 0], dtype=np.float32),
        "aleatorio impar": generar_secuencial(1001),
    }
    for nombre, datos in casos.items():
        original = datos.copy()
        resultado = ordenar(datos)
        assert np.array_equal(resultado, np.sort(original)), nombre
        assert np.array_equal(datos, original), f"{nombre}: modificó la entrada"
    print(f"  ok  ordenamiento ({len(casos)} casos)")


def prueba_histograma_casos_borde():
    iguales = np.full(10, 7.5, dtype=np.float32)
    h = histograma(iguales)
    assert h[0] == 10 and h.sum() == 10, "todos iguales -> cubeta 0"

    uno = np.array([42.0], dtype=np.float32)
    assert histograma(uno).sum() == 1

    # El máximo debe caer en la última cubeta, no perderse
    extremos = np.array([-100.0, 100.0], dtype=np.float32)
    h = histograma(extremos)
    assert h[0] == 1 and h[CUBETAS - 1] == 1, h
    print("  ok  histograma casos borde")


def prueba_histograma_aleatorio(n):
    ordenado = ordenar(generar_secuencial(n))
    h = histograma(ordenado)
    ref = referencia_histograma(ordenado)
    assert h.dtype == np.int32 and len(h) == CUBETAS
    assert h.sum() == n, f"suma {h.sum()} != {n}"
    diferencias = int(np.abs(h - ref).sum())
    assert diferencias == 0, f"{diferencias} elementos en distinta cubeta que la fórmula directa"
    print(f"  ok  histograma N={n:,} (suma = N, coincide con la fórmula de la GPU)")


def prueba_ejecutar(n):
    with tempfile.TemporaryDirectory() as carpeta:
        ordenado, h, tiempos = ejecutar(n, escribir_archivos=True, carpeta_salida=carpeta)

        assert list(tiempos) == FASES, tiempos.keys()
        assert tiempos["h2d"] == 0.0 and tiempos["d2h"] == 0.0
        assert all(tiempos[f] > 0 for f in ("preparacion", "ordenamiento", "histograma", "escritura"))
        suma_fases = sum(v for k, v in tiempos.items() if k != "total")
        assert suma_fases <= tiempos["total"], "el total debe cubrir todas las fases"

        # Mismos datos que la generación compartida
        assert np.array_equal(ordenado, np.sort(generar_secuencial(n)))
        assert h.sum() == n

        with open(os.path.join(carpeta, "temperaturas_ordenadas.csv")) as f:
            lineas = f.read().splitlines()
        assert lineas[0] == "Indice,Temperatura"
        assert len(lineas) == n + 1
        assert lineas[1] == f"0,{ordenado[0]:.2f}"

        with open(os.path.join(carpeta, "histograma.dat")) as f:
            dat = f.read().splitlines()
        assert len(dat) == CUBETAS
        assert sum(int(linea.split()[1]) for linea in dat) == n

        assert os.path.getsize(os.path.join(carpeta, "histograma.png")) > 0
    print(f"  ok  ejecutar N={n:,} (tiempos por fase + CSV, .dat y .png)")


def vista_previa_tiempo(n):
    datos = generar_secuencial(n)
    t0 = time.perf_counter()
    ordenado = ordenar(datos)
    t1 = time.perf_counter()
    histograma(ordenado)
    t2 = time.perf_counter()
    print(f"  ..  N={n:,}: ordenamiento {t1 - t0:.2f} s · histograma {t2 - t1:.3f} s (solo referencia local)")


if __name__ == "__main__":
    print("Pruebas versión secuencial")
    prueba_ordenamiento()
    prueba_histograma_casos_borde()
    for n in (10_000, 100_000):
        prueba_histograma_aleatorio(n)
    prueba_ejecutar(10_000)
    vista_previa_tiempo(1_000_000)
    print("Todo OK")
