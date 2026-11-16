# Paso 6. Comparación secuencial vs CUDA y speedup

> Responsable: Marinés García · Datos: `resultados/tiempos_secuencial.csv`, `resultados/tiempos_cuda.csv`
> → `resultados/speedup.csv` y `resultados/tabla_comparativa.md`
>
> ⚠️ **Antes de entregar:** llenar los ⟨…⟩ con los valores de `resultados/tabla_comparativa.md`.
> La tabla se inserta sola en el informe armado (`python informe/armar_informe.py`).

## 6.1 Condiciones de la comparación

Para que el speedup sea válido, las dos versiones se midieron en las mismas condiciones:

| Condición | Cómo se garantizó |
|---|---|
| Misma máquina | Colab con GPU T4: 1 núcleo físico de CPU (2 lógicos) y 13.6 GB de RAM (`resultados/entorno_*.json`) |
| Mismos datos | Semilla fija (2026) y los mismos 16 bloques de generación. Se verificó que `preparar_datos(n)` produce exactamente el mismo arreglo que `generar_secuencial(n)` |
| Mismo resultado | Antes de medir, el arreglo ordenado en la GPU se comparó con `np.sort` y el histograma de la GPU con el secuencial (`tests/`) |
| Mismos tamaños | N = 10⁴, 10⁵ y 10⁶ |
| Mismas repeticiones | 1 corrida de calentamiento (descartada) + 10 corridas medidas por cada N |
| Mismo trabajo | Las dos versiones generan los datos, ordenan, calculan el histograma y escriben los mismos 3 archivos |
| Transferencias incluidas | El tiempo total de CUDA incluye las copias CPU→GPU (H→D) y GPU→CPU (D→H) |

El **speedup** de cada fase se calcula como:

> Speedup = tiempo secuencial promedio / tiempo CUDA promedio

Un valor mayor que 1× significa que la versión paralela es más rápida; menor que 1×, que es más lenta.

## 6.2 Tabla comparativa de tiempos y speedup por fase

Tiempos promedio de 10 corridas, en segundos. La fila *Cálculo GPU + transferencias* compara el
ordenamiento y el histograma secuenciales contra todo lo que hace la GPU, incluidas las copias de
memoria. El *Límite de Amdahl* es el speedup que se tendría aunque la GPU tardara 0 s: solo
quedarían las fases de CPU de la versión paralela.

<!-- incluir: ../resultados/tabla_comparativa.md -->

## 6.3 Gráficas

![Tiempo total secuencial vs CUDA](figuras/comparativa_total.png)

*Figura 5. Tiempo total de cada versión (escala logarítmica).*

![Speedup por fase](figuras/speedup_por_fase.png)

*Figura 6. Speedup de cada fase. La línea punteada marca 1×: por debajo, la versión CUDA es más lenta.*

![Proporción del tiempo en CUDA](figuras/cuda_porcentaje_por_fase.png)

*Figura 7. En qué se va el tiempo de la versión CUDA. "Otros" es el tiempo que el total mide y
ninguna fase cubre (por ejemplo, leer el mínimo y el máximo o sincronizar con la GPU).*

## 6.4 ¿La paralelización redujo el tiempo de ejecución?

**Speedup total obtenido:**

| N | Secuencial (s) | CUDA (s) | Speedup total | Límite de Amdahl |
|---|---|---|---|---|
| 10,000 | 0.3436 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 100,000 | 0.7303 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 1,000,000 | 6.5672 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

⟨Responder con los números: ¿a partir de qué N la versión CUDA es más rápida? ¿Cuánto más rápida
con N = 10⁶? ¿Qué tan cerca quedó del límite de Amdahl?⟩

**Por fase**, con N = 10⁶:

- **Ordenamiento:** pasó de 5.5106 s a ⟨…⟩ s (⟨…⟩×). Es la fase que más tiempo ahorra en
  términos absolutos, porque era el 83.9 % del tiempo secuencial.
- **Histograma:** pasó de 0.2313 s a ⟨…⟩ s (⟨…⟩×). ⟨Comparar con el speedup del ordenamiento:
  el histograma no tiene pasadas dependientes ni divergencia, así que se espera que acelere más.⟩
- **Transferencias:** H→D ⟨…⟩ s y D→H ⟨…⟩ s, es decir ⟨…⟩ % del tiempo de la GPU. ⟨¿Pesan
  mucho o poco frente a los kernels?⟩
- **Preparación:** ⟨…⟩× ⟨Si es menor que 1×: crear el `Pool` de procesos cuesta más que generar
  los datos, que en el secuencial ya tardaban solo 0.0099 s.⟩
- **Escritura:** ⟨…⟩× ⟨Con 1 núcleo físico, `multiprocessing` solo puede repartir el formateo del
  CSV entre 2 hilos del mismo núcleo.⟩

## 6.5 Causas de las diferencias de rendimiento

⟨Resumir en 1 párrafo, con los números de 6.4. Los factores están desarrollados en la Pregunta 3
(sección siguiente):⟩

- **Por qué CUDA ganó en el cálculo:** miles de hilos clasifican temperaturas y fusionan
  segmentos al mismo tiempo, y los datos se quedan en la GPU entre el ordenamiento y el
  histograma.
- **Por qué el speedup total es mucho menor que el de los kernels:** la escritura de archivos se
  queda en la CPU (Amdahl), y en Colab la CPU tiene un solo núcleo físico.
- **Por qué con N pequeño casi no hay mejora:** los costos fijos (lanzar kernels, transferir,
  crear procesos) pesan tanto como el cálculo, que con N = 10⁴ tardaba solo 0.037 s.
- **Por qué el ordenamiento acelera menos que el histograma:** el MergeSort bottom-up pierde
  paralelismo en las últimas pasadas (en la última, un solo hilo fusiona todo el arreglo).
