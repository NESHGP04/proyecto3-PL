"""Corre las versiones REPETICIONES veces por cada tamaño y guarda los tiempos en CSV.

Responsables: Marinés (--modo seq y --modo graficas) y Carlos (--modo cuda)

Uso (desde la raíz del repo, en Colab):
    python -m benchmarks.run_benchmarks --modo seq
    python -m benchmarks.run_benchmarks --modo cuda
    python -m benchmarks.run_benchmarks --modo graficas

    # pruebas rápidas:
    python -m benchmarks.run_benchmarks --modo seq --tamanos 10000 --repeticiones 3

Salidas:
    resultados/tiempos_<modo>.csv    una fila por corrida
    resultados/resumen_<modo>.csv    promedio y desviación estándar por N y fase
    resultados/entorno_<modo>.json   máquina donde se midió (CPU, Python, GPU)
    informe/figuras/*.png            gráficas (--modo graficas)

Si existen los dos CSV, --modo graficas también genera la comparación (Paso 6):
    resultados/speedup.csv, resultados/tabla_comparativa.md,
    informe/figuras/comparativa_total.png, speedup_por_fase.png, cuda_porcentaje_por_fase.png

Columnas de tiempos_<modo>.csv:
    modo, n, corrida, warmup, preparacion, h2d, ordenamiento, histograma, d2h, escritura, total

En ambos modos, la corrida 0 de cada N es warm-up y se guarda con warmup=1,
pero no entra al promedio:
  - cuda: compilación JIT de Numba y creación del contexto de la GPU
  - seq:  primera importación de matplotlib y cachés frías (en la primera
          prueba local la escritura tardó 0.27 s contra 0.06 s en las demás)
Así las dos versiones se comparan en las mismas condiciones.
"""

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import time

from comun.config import FASES, REPETICIONES, TAMANOS

CARPETA_RESULTADOS = "resultados"
CARPETA_FIGURAS = os.path.join("informe", "figuras")
NOMBRES = {"seq": "secuencial", "cuda": "cuda"}
COLUMNAS = ["modo", "n", "corrida", "warmup"] + FASES
ETIQUETAS_FASES = {
    "preparacion": "Preparación",
    "h2d": "Transferencia H→D",
    "ordenamiento": "Ordenamiento",
    "histograma": "Histograma",
    "d2h": "Transferencia D→H",
    "escritura": "Escritura",
    "total": "Total",
}


# ---------------------------------------------------------------- medición


def obtener_ejecutar(modo):
    """Importa la función ejecutar() de la versión pedida (import tardío:
    el modo seq no necesita Numba ni GPU)."""
    if modo == "seq":
        from secuencial.histograma_seq import ejecutar
    else:
        from paralelo.histograma_cuda import ejecutar
    return ejecutar


def info_entorno(modo):
    """Datos de la máquina, para documentar las condiciones de la medición."""
    info = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modo": modo,
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "procesador": platform.processor() or platform.machine(),
        "nucleos_logicos": os.cpu_count(),
    }
    try:
        import psutil

        info["nucleos_fisicos"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    if modo == "cuda":
        try:
            from numba import cuda

            dispositivo = cuda.get_current_device()
            info["gpu"] = dispositivo.name.decode() if isinstance(dispositivo.name, bytes) else dispositivo.name
            info["compute_capability"] = ".".join(map(str, dispositivo.compute_capability))
        except Exception as e:  # noqa: BLE001  (solo informativo)
            info["gpu"] = f"no detectada: {e}"
    return info


def medir(modo, tamanos, repeticiones):
    """Corre ejecutar() repeticiones veces por cada N y guarda las filas."""
    ejecutar = obtener_ejecutar(modo)
    nombre = NOMBRES[modo]
    carpeta_salida = os.path.join(CARPETA_RESULTADOS, f"salida_{nombre}")
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    with open(os.path.join(CARPETA_RESULTADOS, f"entorno_{nombre}.json"), "w") as f:
        json.dump(info_entorno(modo), f, indent=2, ensure_ascii=False)

    filas = []
    for n in tamanos:
        # Corrida extra (la 0) como warm-up; no entra al promedio
        for corrida in range(repeticiones + 1):
            warmup = int(corrida == 0)
            gc.collect()  # que la basura de la corrida anterior no ensucie esta
            _, hist, tiempos = ejecutar(n, escribir_archivos=True, carpeta_salida=carpeta_salida)
            assert int(hist.sum()) == n, f"histograma inválido en N={n}, corrida {corrida}"

            filas.append({"modo": modo, "n": n, "corrida": corrida, "warmup": warmup, **tiempos})
            etiqueta = " (warm-up)" if warmup else ""
            print(f"  N={n:>10,}  corrida {corrida:>2}  total {tiempos['total']:8.4f} s{etiqueta}")

    ruta = os.path.join(CARPETA_RESULTADOS, f"tiempos_{nombre}.csv")
    with open(ruta, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\nTiempos guardados en {ruta}")

    resumen = resumir(filas)
    guardar_resumen(resumen, nombre)
    imprimir_resumen(resumen, nombre)


# ---------------------------------------------------------------- resumen


def resumir(filas):
    """Promedio y desviación estándar por N y fase, sin contar los warm-up."""
    resumen = []
    for n in sorted({f["n"] for f in filas}):
        validas = [f for f in filas if f["n"] == n and not f["warmup"]]
        fila = {"n": n, "repeticiones": len(validas)}
        for fase in FASES:
            valores = [float(f[fase]) for f in validas]
            fila[f"{fase}_prom"] = statistics.mean(valores)
            fila[f"{fase}_desv"] = statistics.stdev(valores) if len(valores) > 1 else 0.0
            fila[f"{fase}_mediana"] = statistics.median(valores)
        resumen.append(fila)
    return resumen


def guardar_resumen(resumen, nombre):
    ruta = os.path.join(CARPETA_RESULTADOS, f"resumen_{nombre}.csv")
    with open(ruta, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(resumen[0]))
        escritor.writeheader()
        escritor.writerows(resumen)
    print(f"Resumen guardado en {ruta}")


def imprimir_resumen(resumen, nombre):
    print(f"\nPromedios versión {nombre} (segundos, ± desviación estándar)")
    for fila in resumen:
        print(f"\nN = {fila['n']:,}  ({fila['repeticiones']} corridas)")
        for fase in FASES:
            prom, desv = fila[f"{fase}_prom"], fila[f"{fase}_desv"]
            if prom > 0:
                print(f"  {fase:<13} {prom:9.4f} ± {desv:.4f}")


def leer_tiempos(nombre):
    """Lee resultados/tiempos_<nombre>.csv como lista de dicts con números."""
    ruta = os.path.join(CARPETA_RESULTADOS, f"tiempos_{nombre}.csv")
    if not os.path.exists(ruta):
        return None
    with open(ruta) as f:
        filas = list(csv.DictReader(f))
    for fila in filas:
        fila["n"] = int(fila["n"])
        fila["corrida"] = int(fila["corrida"])
        fila["warmup"] = int(fila["warmup"])
        for fase in FASES:
            fila[fase] = float(fila[fase])
    return filas


def tabla_markdown(resumen, fases=("preparacion", "ordenamiento", "histograma", "escritura", "total")):
    """Tablas en Markdown para pegar en el informe:
    1) promedio ± desviación estándar por fase
    2) % del tiempo total y coeficiente de variación del total."""
    lineas = [
        "| N | " + " | ".join(ETIQUETAS_FASES[f] + " (s)" for f in fases) + " |",
        "|---|" + "---|" * len(fases),
    ]
    for fila in resumen:
        celdas = [f"{fila[f'{f}_prom']:.4f} ± {fila[f'{f}_desv']:.4f}" for f in fases]
        lineas.append(f"| {fila['n']:,} | " + " | ".join(celdas) + " |")

    parciales = [f for f in fases if f != "total"]
    lineas += [
        "",
        "| N | " + " | ".join(f"% {ETIQUETAS_FASES[f]}" for f in parciales)
        + " | Mediana del total (s) | CV del total | Corridas |",
        "|---|" + "---|" * (len(parciales) + 3),
    ]
    for fila in resumen:
        total = fila["total_prom"]
        porcentajes = [f"{100 * fila[f'{f}_prom'] / total:.1f} %" for f in parciales]
        cv = 100 * fila["total_desv"] / total
        lineas.append(
            f"| {fila['n']:,} | " + " | ".join(porcentajes)
            + f" | {fila['total_mediana']:.4f} | {cv:.1f} % | {fila['repeticiones']} |"
        )
    return "\n".join(lineas) + "\n"


# ---------------------------------------------------------------- gráficas


def graficas():
    """Genera las figuras del informe a partir de los CSV de tiempos."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    os.makedirs(CARPETA_FIGURAS, exist_ok=True)
    seq = leer_tiempos("secuencial")
    if seq is None:
        print("No existe resultados/tiempos_secuencial.csv: corre primero --modo seq")
        return

    # Se recalcula el resumen desde los tiempos crudos, así siempre
    # corresponde a la versión actual de resumir()
    resumen = resumir(seq)
    guardar_resumen(resumen, "secuencial")
    etiquetas = [f"{f['n']:,}" for f in resumen]
    fases = ["preparacion", "ordenamiento", "histograma", "escritura"]
    colores = ["#8E8E93", "#2E86DE", "#F39C12", "#27AE60"]

    # 1) Tiempo promedio por fase, apilado por tamaño
    fig, ax = plt.subplots(figsize=(9, 5.5))
    base = [0.0] * len(resumen)
    for fase, color in zip(fases, colores):
        valores = [f[f"{fase}_prom"] for f in resumen]
        ax.bar(etiquetas, valores, bottom=base, color=color, label=ETIQUETAS_FASES[fase])
        base = [b + v for b, v in zip(base, valores)]
    for x, total in enumerate(base):
        ax.annotate(f"{total:.3f} s", (x, total), ha="center", va="bottom", fontsize=9)
    ax.set_title("Versión secuencial: tiempo promedio por fase")
    ax.set_xlabel("N (cantidad de temperaturas)")
    ax.set_ylabel("Tiempo (s)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "seq_tiempo_por_fase.png"), dpi=120)
    plt.close(fig)

    # 2) Porcentaje del tiempo total por fase (qué conviene paralelizar)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    base = [0.0] * len(resumen)
    for fase, color in zip(fases, colores):
        valores = [100 * f[f"{fase}_prom"] / f["total_prom"] for f in resumen]
        ax.bar(etiquetas, valores, bottom=base, color=color, label=ETIQUETAS_FASES[fase])
        for x, (b, v) in enumerate(zip(base, valores)):
            if v >= 4:
                ax.text(x, b + v / 2, f"{v:.0f} %", ha="center", va="center", color="white", fontsize=9)
        base = [b + v for b, v in zip(base, valores)]
    ax.set_title("Versión secuencial: proporción del tiempo total por fase")
    ax.set_xlabel("N (cantidad de temperaturas)")
    ax.set_ylabel("% del tiempo total")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "seq_porcentaje_por_fase.png"), dpi=120)
    plt.close(fig)

    # 3) Variabilidad entre corridas del tiempo total (confiabilidad)
    tamanos = [f["n"] for f in resumen]
    fig, axes = plt.subplots(1, len(tamanos), figsize=(max(6, 4 * len(tamanos)), 4.5))
    if len(tamanos) == 1:
        axes = [axes]
    for ax, n in zip(axes, tamanos):
        filas = [f for f in seq if f["n"] == n and not f["warmup"]]
        # La corrida 0 es el warm-up, así que las válidas ya van de 1 a REPETICIONES
        ax.plot([f["corrida"] for f in filas], [f["total"] for f in filas], "o-", color="#2E86DE")
        prom = statistics.mean(f["total"] for f in filas)
        ax.axhline(prom, color="#E74C3C", linestyle="--", label=f"promedio {prom:.3f} s")
        ax.set_title(f"N = {n:,}")
        ax.set_xlabel("Corrida")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Tiempo total (s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Versión secuencial: tiempo total en cada corrida")
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "seq_variabilidad.png"), dpi=120)
    plt.close(fig)

    print(f"Figuras de la versión secuencial guardadas en {CARPETA_FIGURAS}/")

    ruta_tabla = os.path.join(CARPETA_RESULTADOS, "tabla_secuencial.md")
    with open(ruta_tabla, "w") as f:
        f.write(tabla_markdown(resumen))
    print(f"Tabla para el informe guardada en {ruta_tabla}")

    cuda = leer_tiempos("cuda")
    if cuda is None:
        print("Aún no existe resultados/tiempos_cuda.csv: la comparación y el speedup se generan cuando esté.")
        return
    comparar(resumen, resumir(cuda), plt)


# ---------------------------------------------------------------- comparación (Paso 6)

# Filas de la tabla de speedup: (etiqueta, fases secuenciales, fases CUDA).
# El speedup de cada fila es  suma(tiempos seq) / suma(tiempos cuda).
FILAS_SPEEDUP = [
    ("Preparación (CPU)", ["preparacion"], ["preparacion"]),
    ("Ordenamiento (kernel)", ["ordenamiento"], ["ordenamiento"]),
    ("Histograma (kernel)", ["histograma"], ["histograma"]),
    ("Cálculo GPU + transferencias", ["ordenamiento", "histograma"], ["h2d", "ordenamiento", "histograma", "d2h"]),
    ("Escritura (CPU)", ["escritura"], ["escritura"]),
    ("Total", ["total"], ["total"]),
]
LIMITE_AMDAHL = "Límite de Amdahl (GPU en 0 s)"


def calcular_speedup(res_seq, res_cuda):
    """Devuelve una lista de filas {n, fase, seq, cuda, speedup} por cada N común.

    También agrega el límite de Amdahl del pipeline: el speedup que se tendría
    si la parte en GPU (transferencias + kernels) tardara 0 s y solo quedaran
    las fases de CPU que se midieron en la versión paralela:
        S_max = total_seq / (preparacion_cuda + escritura_cuda)
    Ningún kernel más rápido puede superar ese valor.
    """
    por_n_cuda = {f["n"]: f for f in res_cuda}
    filas = []
    for s in res_seq:
        c = por_n_cuda.get(s["n"])
        if c is None:
            continue
        for etiqueta, fases_s, fases_c in FILAS_SPEEDUP:
            t_s = sum(s[f"{f}_prom"] for f in fases_s)
            t_c = sum(c[f"{f}_prom"] for f in fases_c)
            filas.append({"n": s["n"], "fase": etiqueta, "seq": t_s, "cuda": t_c,
                          "speedup": t_s / t_c if t_c > 0 else float("nan")})
        solo_cpu = c["preparacion_prom"] + c["escritura_prom"]
        filas.append({"n": s["n"], "fase": LIMITE_AMDAHL, "seq": s["total_prom"],
                      "cuda": solo_cpu, "speedup": s["total_prom"] / solo_cpu if solo_cpu > 0 else float("nan")})
    return filas


def tabla_comparativa_markdown(filas, res_cuda):
    lineas = []
    for n in sorted({f["n"] for f in filas}):
        lineas += [
            f"**N = {n:,}**",
            "",
            "| Fase | Secuencial (s) | CUDA (s) | Speedup |",
            "|---|---|---|---|",
        ]
        for f in (f for f in filas if f["n"] == n):
            if f["fase"] == LIMITE_AMDAHL:
                lineas.append(f"| *{f['fase']}* | {f['seq']:.4f} | {f['cuda']:.4f} | *{f['speedup']:.2f}×* |")
            else:
                negrita = "**" if f["fase"] == "Total" else ""
                lineas.append(f"| {negrita}{f['fase']}{negrita} | {f['seq']:.4f} | {f['cuda']:.4f} | "
                              f"{negrita}{f['speedup']:.2f}×{negrita} |")
        lineas.append("")

    # Desglose de la versión CUDA con transferencias
    lineas += ["**Desglose de la versión CUDA (promedio ± desviación, s)**", "",
               "| N | " + " | ".join(ETIQUETAS_FASES[f] for f in FASES) + " |",
               "|---|" + "---|" * len(FASES)]
    for c in res_cuda:
        celdas = [f"{c[f'{f}_prom']:.4f} ± {c[f'{f}_desv']:.4f}" for f in FASES]
        lineas.append(f"| {c['n']:,} | " + " | ".join(celdas) + " |")
    return "\n".join(lineas) + "\n"


def comparar(res_seq, res_cuda, plt):
    """Paso 6: speedup por fase, tabla comparativa y gráficas seq vs CUDA."""
    import numpy as np
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    filas = calcular_speedup(res_seq, res_cuda)
    if not filas:
        print("Los CSV de secuencial y CUDA no tienen ningún N en común: no se puede comparar.")
        return

    ruta_csv = os.path.join(CARPETA_RESULTADOS, "speedup.csv")
    with open(ruta_csv, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["n", "fase", "seq", "cuda", "speedup"])
        escritor.writeheader()
        escritor.writerows(filas)

    ruta_md = os.path.join(CARPETA_RESULTADOS, "tabla_comparativa.md")
    with open(ruta_md, "w") as f:
        f.write(tabla_comparativa_markdown(filas, res_cuda))

    tamanos = sorted({f["n"] for f in filas})
    etiquetas = [f"{n:,}" for n in tamanos]
    x = np.arange(len(tamanos))

    def valor(n, fase, campo):
        return next(f[campo] for f in filas if f["n"] == n and f["fase"] == fase)

    # 1) Tiempo total secuencial vs CUDA (escala log: pueden diferir en órdenes de magnitud)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ancho = 0.38
    t_seq = [valor(n, "Total", "seq") for n in tamanos]
    t_cuda = [valor(n, "Total", "cuda") for n in tamanos]
    b1 = ax.bar(x - ancho / 2, t_seq, ancho, label="Secuencial", color="#8E8E93")
    b2 = ax.bar(x + ancho / 2, t_cuda, ancho, label="CUDA", color="#2E86DE")
    ax.bar_label(b1, labels=[f"{t:.3f} s" for t in t_seq], fontsize=8)
    ax.bar_label(b2, labels=[f"{t:.3f} s" for t in t_cuda], fontsize=8)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x, etiquetas)
    ax.set_title("Tiempo total: secuencial vs CUDA")
    ax.set_xlabel("N (cantidad de temperaturas)")
    ax.set_ylabel("Tiempo (s, escala logarítmica)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "comparativa_total.png"), dpi=120)
    plt.close(fig)

    # 2) Speedup por fase para cada N (línea en 1×: debajo, CUDA es más lento)
    fases_graf = [e for e, _, _ in FILAS_SPEEDUP]
    colores = ["#8E8E93", "#2E86DE", "#F39C12", "#9B59B6", "#27AE60", "#E74C3C"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ancho = 0.8 / len(fases_graf)
    for k, (fase, color) in enumerate(zip(fases_graf, colores)):
        valores = [valor(n, fase, "speedup") for n in tamanos]
        barras = ax.bar(x + (k - (len(fases_graf) - 1) / 2) * ancho, valores, ancho, label=fase, color=color)
        ax.bar_label(barras, labels=[f"{v:.1f}×" if v >= 1 else f"{v:.2f}×" for v in valores],
                     fontsize=7, rotation=90, padding=2)
    ax.axhline(1, color="black", linewidth=1, linestyle="--")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xticks(x, etiquetas)
    ax.set_title("Speedup por fase (secuencial / CUDA)")
    ax.set_xlabel("N (cantidad de temperaturas)")
    ax.set_ylabel("Speedup (escala logarítmica)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.margins(y=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "speedup_por_fase.png"), dpi=120)
    plt.close(fig)

    # 3) En qué se va el tiempo de la versión CUDA (incluye transferencias)
    por_n = {c["n"]: c for c in res_cuda}
    fases_cuda = ["preparacion", "h2d", "ordenamiento", "histograma", "d2h", "escritura"]
    colores_cuda = ["#8E8E93", "#C0392B", "#2E86DE", "#F39C12", "#E67E22", "#27AE60"]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    base = np.zeros(len(tamanos))
    series = [(ETIQUETAS_FASES[f], c, [por_n[n][f"{f}_prom"] for n in tamanos])
              for f, c in zip(fases_cuda, colores_cuda)]
    # Lo que el total mide y ninguna fase cubre (p. ej. leer min/max, sincronizar)
    otros = [max(0.0, por_n[n]["total_prom"] - sum(por_n[n][f"{f}_prom"] for f in fases_cuda)) for n in tamanos]
    series.append(("Otros (sin medir)", "#D5D8DC", otros))
    for etiqueta, color, tiempos in series:
        valores = np.array([100 * t / por_n[n]["total_prom"] for t, n in zip(tiempos, tamanos)])
        ax.bar(etiquetas, valores, bottom=base, color=color, label=etiqueta)
        for i, (b, v) in enumerate(zip(base, valores)):
            if v >= 4:
                ax.text(i, b + v / 2, f"{v:.0f} %", ha="center", va="center",
                        color="black" if color == "#D5D8DC" else "white", fontsize=9)
        base += valores
    ax.set_title("Versión CUDA: proporción del tiempo total por fase")
    ax.set_xlabel("N (cantidad de temperaturas)")
    ax.set_ylabel("% del tiempo total")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "cuda_porcentaje_por_fase.png"), dpi=120)
    plt.close(fig)

    print(f"Speedup guardado en {ruta_csv} y {ruta_md}")
    print(f"Figuras de comparación guardadas en {CARPETA_FIGURAS}/")
    for n in tamanos:
        print(f"  N = {n:>10,}: speedup total {valor(n, 'Total', 'speedup'):.2f}×  "
              f"(límite de Amdahl {valor(n, LIMITE_AMDAHL, 'speedup'):.2f}×)")


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description="Benchmarks del Proyecto 3")
    p.add_argument("--modo", choices=["seq", "cuda", "graficas"], required=True)
    p.add_argument("--tamanos", type=int, nargs="+", default=TAMANOS, help="valores de N")
    p.add_argument("--repeticiones", type=int, default=REPETICIONES)
    args = p.parse_args()

    if args.modo == "graficas":
        graficas()
    else:
        print(f"Benchmark modo {args.modo}: N = {args.tamanos}, {args.repeticiones} repeticiones\n")
        medir(args.modo, args.tamanos, args.repeticiones)


if __name__ == "__main__":
    main()
