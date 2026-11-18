# Paso 5. Implementación con CUDA Python

**Responsable: Carlos Alburez (231311).** Proyecto 3: histograma y MergeSort con CUDA.

## 1. Arquitectura del Pipeline Híbrido CPU-GPU

La solución paralela implementa una arquitectura por fases (`paralelo/histograma_cuda.py`) que acopla el procesamiento multinúcleo en CPU para operaciones de entrada/salida y preparación con la capacidad masiva de cálculo de la GPU NVIDIA T4 en Google Colab:

| Fase | Componente | Tecnología | Ubicación | Medición |
|---|---|---|---|---|
| **F1. Preparación de datos** | Generación de $N$ temperaturas en bloques | `multiprocessing.Pool.starmap` | CPU | `time.perf_counter()` |
| **F2. Transferencia H→D** | Copia de arreglo `float32` a VRAM | `cuda.to_device` | Bus PCIe | `cuda.event()` |
| **F3. Ordenamiento** | MergeSort bottom-up por pasadas | Kernel Numba `@cuda.jit` | GPU (VRAM) | `cuda.event()` |
| **Extremos** | Lectura de mínimo y máximo (1 float32 c/u) | Slice `:1` y `-1:` a host | Host/GPU | Integrado en F4/Total |
| **F4. Histograma** | 100 cubetas con memoria compartida | Kernel Numba con atómicos | GPU (Shared/VRAM) | `cuda.event()` |
| **F5. Transferencia D→H** | Copia de arreglo ordenado e histograma | `DeviceNDArray.copy_to_host` | Bus PCIe | `cuda.event()` |
| **F6. Escritura de resultados** | Formateo paralelo y escritura en orden | `Pool.imap` + Proceso padre | CPU / Disco | `time.perf_counter()` |

### Retención de datos en memoria del dispositivo (Zero-Copy Intermedio)
Un factor crítico de diseño es que **el arreglo de temperaturas permanece en la VRAM de la GPU entre el ordenamiento (F3) y el histograma (F4)**. Dado que $N = 10^6$ en `float32` ocupa 4 MB, transferir el arreglo de regreso a la CPU tras ordenarlo y volver a enviarlo a la GPU para el histograma hubiera introducido una penalización redundante de ancho de banda en el bus PCIe. El pipeline ejecuta ambas fases en el dispositivo y únicamente transfiere al host los escalares del mínimo y máximo (`ordenado[0]` y `ordenado[-1]`) para configurar el ancho de cubeta.

---

## 2. Kernel MergeSort bottom-up en CUDA (`paralelo/kernel_mergesort.py`)

La versión secuencial (`secuencial/histograma_seq.py`) utilizaba recursión directa (`merge_sort`). Trasladar recursión a la GPU resulta ineficiente debido a la limitada pila de ejecución por hilo y la incapacidad de generar paralelismo masivo a partir de llamadas recursivas bifurcadas.

En su lugar, se implementó el **MergeSort bottom-up iterativo**:
1. Se realizan pasadas sucesivas con anchos crecientes de subarreglo $w \in \{1, 2, 4, 8, \dots\}$, duplicando el ancho en cada iteración mientras $w < N$.
2. En cada pasada, se lanza una cuadrícula de hilos donde cada hilo $t$ es responsable de fusionar dos segmentos contiguos ordenados:
   $$\text{inicio} = t \times (2w), \quad \text{medio} = \min(\text{inicio} + w, N), \quad \text{fin} = \min(\text{inicio} + 2w, N)$$
3. **Esquema de buffers alternados (Ping-Pong):** Se reserva un único buffer auxiliar `d_aux = cuda.device_array_like(d_datos)` en la GPU. En cada pasada, el kernel lee de `src` y escribe ordenado en `dst`, alternando las referencias (`src, dst = dst, src`) al terminar la pasada. Al completar las ⌈log₂ N⌉ pasadas (20 pasadas para N = 1,000,000), el resultado final queda en la VRAM listo para la siguiente fase, sin asignaciones dinámicas ni transferencias intermedias.

```python
@cuda.jit
def kernel_merge_pasada(src, dst, ancho, n, num_merges):
    tid = cuda.grid(1)
    salto = cuda.gridsize(1)
    for t in range(tid, num_merges, salto):
        inicio = t * (2 * ancho)
        medio = min(inicio + ancho, n)
        fin = min(inicio + 2 * ancho, n)
        
        i, j, k = inicio, medio, inicio
        while i < medio and j < fin:
            if src[i] <= src[j]:
                dst[k] = src[i]; i += 1
            else:
                dst[k] = src[j]; j += 1
            k += 1
        while i < medio:
            dst[k] = src[i]; i += 1; k += 1
        while j < fin:
            dst[k] = src[j]; j += 1; k += 1
```

### Configuración del Grid y Análisis de Divergencia
- Se fijan **$T = 256$ hilos por bloque**.
- Para cada pasada con ancho $w$, la cantidad de fusiones independientes requeridas es:
  $$\text{num\_merges} = \left\lceil \frac{N}{2w} \right\rceil$$
- La cantidad de bloques lanzados se acota mediante:
  $$B = \min\left(\left\lceil \frac{\text{num\_merges}}{256} \right\rceil, 65535\right)$$
- **Comportamiento dinámico del paralelismo:**
  - En las primeras pasadas ($w=1, 2, 4$), existen cientos de miles de fusiones independientes, logrando saturar todos los SM de la GPU T4 (2,560 núcleos CUDA).
  - En las últimas pasadas ($w \ge N/2$), el número de fusiones activas decrece drásticamente. En la última pasada ($\text{num\_merges} = 1$), **un único hilo** fusiona los dos bloques de $N/2$ elementos mientras los demás hilos del warp divergen. A pesar de esta disminución en el paralelismo hacia el final, el costo total sigue siendo considerablemente inferior al de la CPU gracias a la velocidad de la memoria de la GPU y la reducción masiva en las primeras fases.

---

## 3. Instrumentación de Tiempos y Monitoreo de Recursos

### Medición precisa mediante `cuda.event`
Las llamadas a kernels de CUDA son asíncronas respecto al host. Medir el tiempo de GPU con temporizadores de CPU (`time.perf_counter()`) sin barreras introduce errores de medición por latencia de despacho. 

Por ello, el pipeline utiliza eventos de hardware de CUDA (`cuda.event(timing=True)`):
```python
ev_inicio = cuda.event()
ev_fin = cuda.event()

ev_inicio.record()
d_ordenado = mergesort_gpu(d_datos)
ev_fin.record()
ev_fin.synchronize()
tiempos["ordenamiento"] = cuda.event_elapsed_time(ev_inicio, ev_fin) / 1000.0
```
Esto garantiza registrar estrictamente el tiempo de ejecución en silicio de la GPU y el ancho de banda del bus PCIe, libre de interferencias del sistema operativo.

### Monitoreo de Hardware (`benchmarks/monitor_recursos.py`)
Para cumplir con los 30 puntos de evidencia de la rúbrica:
1. **Monitor de CPU en segundo plano (`MonitorCPU`):** Un hilo desacoplado muestrea `psutil.cpu_percent(percpu=True)` a intervalos regulares (0.2 a 0.5 s) durante la ejecución, guardando las lecturas en `evidencias/cpu_por_nucleo.csv` para constatar la distribución multinúcleo en las fases F1 y F6.
2. **Monitoreo de GPU (`nvidia-smi` y `dmon`):** Se registran volcados de estado antes y durante las ejecuciones (`evidencias/nvidia_smi_*.txt`), así como el log continuo de métricas de SM y memoria con `nvidia-smi dmon -s um -d 1 > evidencias/gpu_dmon.log`.

---

## 4. Validación Obligatoria y Reproducibilidad

La función `validar(n)` implementada en `paralelo/histograma_cuda.py` se ejecuta antes de cualquier medición de rendimiento y verifica tres condiciones de parada:
1. **Identidad con `np.sort`:** `np.testing.assert_allclose(ordenado, np.sort(datos_ref), rtol=1e-5)` comprueba que el arreglo ordenado por CUDA sea idéntico al algoritmo estándar de referencia.
2. **Identidad con el Histograma Secuencial:** `np.testing.assert_array_equal(hist, hist_secuencial)` comprueba que las frecuencias de las 100 cubetas coincidan valor por valor con la línea base de Marinés.
3. **Conservación de masa:** $\sum_{c=0}^{99} \text{hist}[c] = N$, garantizando que ningún valor quede fuera de rango ni se duplique por condiciones de carrera.
