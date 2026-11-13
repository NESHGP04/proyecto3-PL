"""Corre las versiones REPETICIONES veces por cada tamaño y guarda los tiempos en CSV.

Responsables: Marinés (--modo seq) y Carlos (--modo cuda)

Uso (desde la raíz del repo, en Colab):
    python -m benchmarks.run_benchmarks --modo seq
    python -m benchmarks.run_benchmarks --modo cuda
    python -m benchmarks.run_benchmarks --modo graficas   # tabla + speedup por fase

Salida: resultados/tiempos_secuencial.csv, resultados/tiempos_cuda.csv
Columnas: modo, n, corrida, preparacion, h2d, ordenamiento, histograma, d2h, escritura, total
En modo cuda, la corrida 0 es warm-up (compilación JIT): se guarda, pero no entra al promedio.
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modo", choices=["seq", "cuda", "graficas"], required=True)
    args = p.parse_args()
    raise NotImplementedError(f"TODO modo {args.modo}")


if __name__ == "__main__":
    main()
