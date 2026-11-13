# Proyecto 3: Histograma + MergeSort con CUDA

CC3069 Computación Paralela y Distribuida, UVG 2026.
Marinés García · Camila Richter · Carlos Alburez

Paralelizamos el programa de histograma de temperaturas (MergeSort + 100 cubetas) de la
Evaluación Corta 2. La preparación de datos y los archivos se procesan en el **CPU multinúcleo**
y el ordenamiento y el histograma corren en la **GPU con CUDA (Numba)**.

📋 **División de tareas, fechas y acuerdos técnicos: [PLAN.md](PLAN.md)**

## Correr en Google Colab

1. *Entorno de ejecución → Cambiar tipo de entorno → T4 GPU*
2. En una celda:

```python
!git clone https://github.com/NESHGP04/proyecto3-PL.git
%cd proyecto3-PL
!pip install -q -r requirements.txt
from numba import cuda; cuda.detect()
```

3. Benchmarks:

```python
!python -m benchmarks.run_benchmarks --modo seq
!python -m benchmarks.run_benchmarks --modo cuda
!python -m benchmarks.run_benchmarks --modo graficas
```

> Si `from numba import cuda` falla o muestra un aviso de que está obsoleto, correr `!pip install numba-cuda`.
