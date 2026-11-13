"""Evidencia de uso de hardware durante las corridas (30 pts de la rúbrica).

Responsable: Carlos (rama carlos/mergesort-gpu)

- GPU: lanzar en segundo plano antes del benchmark:
      nvidia-smi dmon -s um -d 1 > evidencias/gpu_dmon.log &
  y guardar `nvidia-smi` antes y durante la corrida en evidencias/nvidia_smi_*.txt
- CPU: hilo que muestrea psutil.cpu_percent(percpu=True) cada 0.5 s y lo guarda en
  evidencias/cpu_por_nucleo.csv, para mostrar la carga por núcleo en la fase F1/F6.
- Capturas: panel "Recursos" de Colab durante cada fase, en evidencias/*.png
"""


class MonitorCPU:
    """Context manager: with MonitorCPU("evidencias/cpu_por_nucleo.csv"): ..."""

    def __init__(self, ruta_csv, intervalo=0.5):
        raise NotImplementedError("TODO Carlos")
