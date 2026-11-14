"""Pruebas de la contribución de Camila; no ejecutan MergeSort ni benchmarks."""
import csv
import importlib.util
import multiprocessing
import tempfile
import unittest
from pathlib import Path

import numpy as np

from comun.config import CUBETAS
from comun.datos import generar_secuencial
from paralelo.preparacion_cpu import preparar_datos, escribir_resultados


def referencia_histograma(datos):
    """Oráculo independiente: buscar límites sobre datos, sin fórmula de cociente."""
    datos = np.asarray(datos, dtype=np.float32)
    if not datos.size:
        return np.zeros(CUBETAS, dtype=np.int32)
    minimo, maximo = float(datos.min()), float(datos.max())
    if minimo == maximo:
        hist = np.zeros(CUBETAS, dtype=np.int32)
        hist[0] = datos.size
        return hist
    ancho = (maximo - minimo) / CUBETAS
    interiores = minimo + np.arange(1, CUBETAS, dtype=np.float64) * ancho
    indices = np.searchsorted(interiores, datos.astype(np.float64), side="right")
    return np.bincount(indices, minlength=CUBETAS).astype(np.int32)


class PruebasCPU(unittest.TestCase):
    def test_mismos_datos_distintos_procesos(self):
        for n in (1, 7, 257, 10_000):
            for procesos in (1, 2):
                with self.subTest(n=n, procesos=procesos):
                    datos = preparar_datos(n, procesos)
                    self.assertEqual(datos.dtype, np.float32)
                    np.testing.assert_array_equal(datos, generar_secuencial(n))

    def test_preparacion_vacia(self):
        datos = preparar_datos(0, procesos=2)
        self.assertEqual(datos.shape, (0,))
        self.assertEqual(datos.dtype, np.float32)

    def test_argumentos_invalidos(self):
        with self.assertRaises(ValueError):
            preparar_datos(-1)
        with self.assertRaises(ValueError):
            preparar_datos(10, procesos=0)
        with self.assertRaises(TypeError):
            preparar_datos(1.5)

    def test_archivos_ordenados_y_frecuencias(self):
        ordenado = np.sort(generar_secuencial(1003))
        hist = referencia_histograma(ordenado)
        with tempfile.TemporaryDirectory() as carpeta:
            rutas = escribir_resultados(ordenado, hist, carpeta, procesos=2)
            with rutas["temperaturas_ordenadas.csv"].open() as archivo:
                filas = list(csv.reader(archivo))
            self.assertEqual(filas[0], ["Indice", "Temperatura"])
            self.assertEqual(len(filas), ordenado.size + 1)
            self.assertEqual([int(f[0]) for f in filas[1:]], list(range(ordenado.size)))
            np.testing.assert_allclose(
                [float(f[1]) for f in filas[1:]], ordenado, rtol=0, atol=0.00001
            )
            dat = np.loadtxt(rutas["histograma.dat"])
            self.assertEqual(dat.shape, (CUBETAS, 2))
            np.testing.assert_array_equal(dat[:, 1], hist)
            self.assertEqual(rutas["histograma.png"].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            # Distinto número de workers conserva los archivos de texto exactos.
            otros = escribir_resultados(ordenado, hist, Path(carpeta) / "uno", procesos=1)
            for nombre in ("temperaturas_ordenadas.csv", "histograma.dat"):
                self.assertEqual(rutas[nombre].read_bytes(), otros[nombre].read_bytes())

    def test_archivos_vacios_y_constantes(self):
        for datos in (np.empty(0, np.float32), np.full(19, 3.5, np.float32)):
            with self.subTest(n=datos.size), tempfile.TemporaryDirectory() as carpeta:
                rutas = escribir_resultados(datos, referencia_histograma(datos), carpeta, 2)
                self.assertTrue(all(p.exists() for p in rutas.values()))
                self.assertEqual(np.loadtxt(rutas["histograma.dat"])[:, 1].sum(), datos.size)

    def test_rechazar_histograma_invalido(self):
        with tempfile.TemporaryDirectory() as carpeta:
            with self.assertRaises(ValueError):
                escribir_resultados(np.ones(2, np.float32), np.zeros(CUBETAS, np.int32), carpeta)


class PruebasCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("numba") is None:
            raise unittest.SkipTest("Numba no está instalado; ejecutar en Colab")
        from numba import cuda
        if not cuda.is_available():
            raise unittest.SkipTest("CUDA no disponible; usar GPU o NUMBA_ENABLE_CUDASIM=1")
        from paralelo.kernel_histograma import histograma_gpu, kernel_histograma
        cls.cuda = cuda
        cls.histograma_gpu = staticmethod(histograma_gpu)
        cls.kernel_histograma = staticmethod(kernel_histograma)

    def comprobar(self, datos):
        ordenado = np.sort(np.asarray(datos, dtype=np.float32))
        minimo = float(ordenado[0]) if ordenado.size else 0.0
        maximo = float(ordenado[-1]) if ordenado.size else 0.0
        d_datos = self.cuda.to_device(ordenado)
        hist = self.histograma_gpu(d_datos, minimo, maximo).copy_to_host()
        self.assertEqual(hist.dtype, np.int32)
        self.assertEqual(int(hist.sum()), ordenado.size)
        np.testing.assert_array_equal(hist, referencia_histograma(ordenado))
        return ordenado, hist

    def test_casos_limite_y_fronteras(self):
        fronteras = np.arange(-100, 101, 2, dtype=np.float32)
        casos = [[], [4.0], [3.5] * 129, [-100, 100], fronteras,
                 np.concatenate((np.nextafter(fronteras[1:-1], np.float32(-np.inf)),
                                 fronteras, np.nextafter(fronteras[1:-1], np.float32(np.inf))))]
        for datos in casos:
            with self.subTest(n=len(datos)):
                self.comprobar(datos)

    def test_datos_reales_y_reinicializacion(self):
        # N no múltiplo de 256; dos llamadas deben producir la misma cuenta.
        for _ in range(2):
            self.comprobar(generar_secuencial(1025))

    def test_grid_stride_y_bloque_menor_que_100(self):
        datos = generar_secuencial(2049)
        minimo, maximo = float(datos.min()), float(datos.max())
        hist = self.cuda.to_device(np.zeros(CUBETAS, np.int32))
        self.kernel_histograma[2, 32](self.cuda.to_device(datos), minimo,
                                    (maximo - minimo) / CUBETAS, hist)
        np.testing.assert_array_equal(hist.copy_to_host(), referencia_histograma(datos))

    def test_comparacion_con_secuencial_del_equipo(self):
        from secuencial.histograma_seq import histograma
        ordenado, obtenido = self.comprobar(generar_secuencial(513))
        try:
            esperado = histograma(ordenado)
        except NotImplementedError:
            self.skipTest("La función secuencial del equipo todavía está pendiente")
        np.testing.assert_array_equal(obtenido, esperado)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    unittest.main(verbosity=2)
