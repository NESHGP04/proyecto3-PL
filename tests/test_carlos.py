
import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path al correr directamente como script
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

try:
    import numpy as np
except ImportError:
    np = None

from comun.config import CUBETAS, FASES
from comun.datos import generar_secuencial
from benchmarks.monitor_recursos import MonitorCPU


class PruebasCarlosCPU(unittest.TestCase):
    """Pruebas que no requieren GPU ni CUDA."""

    def test_monitor_cpu(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta_csv = f"{carpeta}/cpu_test.csv"
            with MonitorCPU(ruta_csv, intervalo=0.1):
                time.sleep(0.35)

            # Verificar que el CSV se creó y tiene datos
            with open(ruta_csv, "r") as f:
                lineas = f.readlines()
            self.assertGreaterEqual(len(lineas), 2, "El monitor debe registrar al menos el encabezado y una muestra")
            self.assertIn("timestamp", lineas[0])
            self.assertIn("cpu_0", lineas[0])


class PruebasCarlosGPU(unittest.TestCase):
    """Pruebas de kernels CUDA y pipeline en GPU (se omiten si no hay GPU/CUDA)."""

    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("numba") is None:
            raise unittest.SkipTest("Numba no está instalado; ejecutar en Colab")
        from numba import cuda
        if not cuda.is_available():
            raise unittest.SkipTest("CUDA no disponible en este entorno; usar GPU o NUMBA_ENABLE_CUDASIM=1")
        from paralelo.kernel_mergesort import mergesort_gpu
        import paralelo.histograma_cuda as cuda_pipe
        cls.cuda = cuda
        cls.mergesort_gpu = staticmethod(mergesort_gpu)
        cls.cuda_pipe = cuda_pipe

    def test_mergesort_casos_borde(self):
        casos = [
            np.empty(0, dtype=np.float32),
            np.array([42.0], dtype=np.float32),
            np.array([5.0, -2.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([9.0, 7.0, 5.0, 3.0, 1.0], dtype=np.float32),
            np.full(16, 3.14, dtype=np.float32),
            np.array([4.0, 1.0, 4.0, 2.0, 1.0, 4.0, 3.0], dtype=np.float32),
        ]
        for datos in casos:
            with self.subTest(n=len(datos)):
                d_datos = self.cuda.to_device(datos)
                d_ordenado = self.mergesort_gpu(d_datos)
                obtenido = d_ordenado.copy_to_host()
                esperado = np.sort(datos)
                np.testing.assert_array_equal(obtenido, esperado)

    def test_mergesort_tamanos_variados(self):
        for n in (7, 64, 255, 256, 1025, 10_000):
            with self.subTest(n=n):
                datos = generar_secuencial(n)
                d_datos = self.cuda.to_device(datos)
                d_ordenado = self.mergesort_gpu(d_datos)
                obtenido = d_ordenado.copy_to_host()
                esperado = np.sort(datos)
                np.testing.assert_array_equal(obtenido, esperado)

    def test_pipeline_cuda_completo(self):
        n = 1000
        ordenado, hist, tiempos = self.cuda_pipe.ejecutar(n, escribir_archivos=False)
        self.assertEqual(len(ordenado), n)
        self.assertEqual(len(hist), CUBETAS)
        self.assertEqual(int(hist.sum()), n)
        for fase in FASES:
            self.assertIn(fase, tiempos)
            self.assertGreaterEqual(tiempos[fase], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
