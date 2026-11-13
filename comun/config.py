"""Configuración compartida por la versión secuencial y la paralela.

No cambiar estos valores sin avisar al grupo: los benchmarks dependen de ellos.
"""

# Histograma
CUBETAS = 100
TEMP_MIN = -100.0
TEMP_MAX = 100.0

# Reproducibilidad: misma semilla => mismos datos en ambas versiones
SEMILLA = 2026

# Benchmarks
TAMANOS = [10_000, 100_000, 1_000_000]
REPETICIONES = 10

# Los datos se generan en este número fijo de bloques, tanto en el secuencial
# (loop) como en el paralelo (Pool). Así el arreglo resultante es idéntico
# sin importar cuántos núcleos tenga la máquina.
BLOQUES_DATOS = 16

# CUDA
HILOS_POR_BLOQUE = 256

# Llaves del diccionario de tiempos que devuelve cada versión (en segundos)
FASES = ["preparacion", "h2d", "ordenamiento", "histograma", "d2h", "escritura", "total"]
