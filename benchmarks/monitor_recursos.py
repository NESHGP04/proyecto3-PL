

import csv
import os
import shutil
import subprocess
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None


class MonitorCPU:
    """Context manager para registrar la carga por núcleo de CPU en segundo plano.

    Uso:
        with MonitorCPU("evidencias/cpu_por_nucleo.csv", intervalo=0.5):
            ejecutar(1_000_000)
    """

    def __init__(self, ruta_csv, intervalo=0.5):
        self.ruta_csv = ruta_csv
        self.intervalo = intervalo
        self._detener = threading.Event()
        self._hilo = None

    def __enter__(self):
        if psutil is None:
            return self
        os.makedirs(os.path.dirname(self.ruta_csv) or ".", exist_ok=True)
        self._detener.clear()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()
        return self

    def _muestrear(self):
        num_nucleos = psutil.cpu_count(logical=True) or 1
        encabezados = ["timestamp", "segundo_relativo"] + [f"cpu_{i}" for i in range(num_nucleos)]
        t0 = time.time()
        with open(self.ruta_csv, "w", newline="") as f:
            escritor = csv.writer(f)
            escritor.writerow(encabezados)
            # Llamada inicial de calentamiento para psutil
            psutil.cpu_percent(percpu=True)
            while not self._detener.is_set():
                time.sleep(self.intervalo)
                cargas = psutil.cpu_percent(percpu=True)
                ahora = time.time()
                fila = [f"{ahora:.3f}", f"{ahora - t0:.2f}"] + [f"{c:.1f}" for c in cargas]
                escritor.writerow(fila)
                f.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hilo and self._hilo.is_alive():
            self._detener.set()
            self._hilo.join(timeout=2.0)


def guardar_estado_gpu(ruta_salida):
    """Ejecuta nvidia-smi y guarda el volcado de estado en un archivo."""
    os.makedirs(os.path.dirname(ruta_salida) or ".", exist_ok=True)
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            with open(ruta_salida, "w") as f:
                f.write(res.stdout)
            print(f"Estado de GPU guardado en {ruta_salida}")
        except Exception as e:
            print(f"Aviso: no se pudo ejecutar nvidia-smi: {e}")
    else:
        print("Aviso: comando nvidia-smi no disponible en este entorno.")


def iniciar_monitoreo_dmon(ruta_salida="evidencias/gpu_dmon.log"):
    """Inicia nvidia-smi dmon en segundo plano si está disponible."""
    os.makedirs(os.path.dirname(ruta_salida) or ".", exist_ok=True)
    if shutil.which("nvidia-smi"):
        try:
            archivo = open(ruta_salida, "w")
            proc = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "um", "-d", "1"],
                stdout=archivo,
                stderr=subprocess.DEVNULL
            )
            return proc, archivo
        except Exception as e:
            print(f"Aviso: no se pudo iniciar nvidia-smi dmon: {e}")
    return None, None
