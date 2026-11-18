# Paso 3. Oportunidades de paralelización

**Responsable: Camila Richter (23183).** Proyecto 3: histograma y MergeSort con CUDA.

## Operaciones y decisiones

| Operación | Reparto propuesto | Justificación y límites |
|---|---|---|
| Generar temperaturas, F1 | CPU: `Pool.starmap(generar_bloque, semillas_bloques(n))` | Cada bloque tiene una semilla propia y no depende de otros. Los trabajadores pueden producirlos simultáneamente. Crear procesos, serializar y concatenar agrega costo; para N pequeño podría ser más lento. |
| Ordenar, F3 | GPU: fusiones independientes de MergeSort bottom-up (implementado por Carlos) | Dentro de una pasada las parejas de segmentos no se superponen. La pasada siguiente necesita la anterior. |
| Contar frecuencias, F4 | GPU: un histograma compartido por bloque y suma al global | Cada temperatura puede clasificarse de manera independiente. Los contadores sí se comparten; se protegen con sumas atómicas. |
| Formatear CSV, F6 | CPU: bloques con `Pool.imap`, manteniendo el orden | Convertir cada temperatura a texto es independiente. Cada trabajador devuelve una cadena y el padre escribe las cadenas en su orden original. |
| Escribir físicamente el CSV | Un único proceso padre | Evita líneas intercaladas y mantiene los índices. El almacenamiento y el acceso al mismo archivo limitan el paralelismo útil. |
| Escribir `.dat` y dibujar `.png` | CPU, proceso padre | Solo hay 100 frecuencias. Crear procesos para estas salidas pequeñas agregaría más coordinación que trabajo útil. |
| Transferir CPU↔GPU | Operaciones del pipeline (implementado por Carlos) | Son costos necesarios, no aceleración matemática. F2 y F5 están incluidos en el tiempo total. |

## Reparto estático y ejecución dinámica

La preparación conserva los **16 bloques lógicos** definidos en `comun/config.py`, o menos si N es menor que 16. Los tamaños y semillas se fijan antes del `Pool`: es una partición estática, equilibrada con una diferencia máxima de un elemento. El Pool entrega las tareas a sus trabajadores disponibles; `starmap` devuelve los resultados en el orden de entrada, independientemente de su orden de finalización. Cambiar el número de procesos no cambia los datos.

En F6 se forman bloques de hasta 65 536 temperaturas, aproximadamente 16 para los tamaños habituales. Los trabajadores formatean bloques independientes. `imap(..., chunksize=1)` entrega los textos en el orden original; solo el padre escribe el archivo. Para N grande pueden existir más de 16 tareas, con un máximo de 16 trabajadores.

En GPU el tamaño del grid y los índices se definen antes del lanzamiento. El índice inicial del hilo es `i = blockIdx.x * blockDim.x + threadIdx.x` y el salto es `gridDim.x * blockDim.x`. Cada índice pertenece a un único hilo. El dispositivo programa los bloques disponibles; no hay una cola de tareas implementada por el equipo.

## Qué permanece secuencial y cómo se reduce la coordinación

No se intenta trasladar directamente la recursión del MergeSort de Python al kernel. El diseño del equipo usa pasadas bottom-up y conserva la dependencia entre ellas. Tampoco se paraleliza la escritura concurrente al mismo archivo: se paraleliza su formateo y se conserva un escritor.

El histograma evita una barrera por temperatura. Cada bloque ejecuta solamente dos barreras: después de inicializar sus contadores y después de contarlos. La fusión de histogramas usa atómicos globales, sin barrera entre bloques. El arreglo permanece en el dispositivo entre ordenamiento e histograma; la función de Camila devuelve el histograma en GPU y deja la copia al host al pipeline.

## Condiciones para una comparación justa

- Comparar `preparar_datos(n)` con `comun.datos.generar_secuencial(n)` elemento por elemento, usando los mismos bloques y semillas.
- Incluir creación/cierre del Pool, comunicación entre procesos y concatenación en F1; incluir formateo, escritura y gráfica en F6.
- Inicializar el histograma global en cada llamada para no acumular resultados anteriores.
- Verificar la suma de frecuencias y las cubetas antes de medir. No basta con que `sum(hist) == N`.
- Descartar la corrida de compilación JIT según el plan. La simulación de CUDA no sirve para medir speedup.

El análisis cuantitativo de tiempos, mediciones experimentales y el cálculo de speedup por fase se presentan de forma integral en el Paso 6, tras la ejecución de los benchmarks oficiales del pipeline completo en Google Colab con la GPU NVIDIA T4.

## Referencias técnicas

- [API de kernels de Numba CUDA: memoria compartida, barreras y atómicos](https://nvidia.github.io/numba-cuda/reference/kernel.html).
- [Simulador CUDA y sus límites](https://nvidia.github.io/numba-cuda/user/simulator.html).
- Código compartido del equipo: `comun/config.py` y `comun/datos.py`, conservados sin cambios.
