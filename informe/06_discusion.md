# Paso 6. Discusión técnica: aporte de Camila

**Camila Richter (23183).** 

## Pregunta 1. ¿Por qué conviene procesar archivos en CPU multinúcleo y delegar solo la matemática a la GPU?

La CPU es adecuada para coordinar procesos, generar bloques de entrada, convertir números a texto y acceder al sistema de archivos. En nuestra propuesta, `multiprocessing.Pool` reparte la generación de temperaturas y el formateo del CSV entre los núcleos disponibles. Cada proceso trabaja con un bloque independiente y el proceso principal escribe los textos en orden, evitando que se mezclen las filas del archivo.

La GPU se aprovecha cuando muchos datos requieren operaciones similares. Por eso recibe el ordenamiento con MergeSort y el conteo de las 100 cubetas. Para el histograma, cada bloque cuenta primero en memoria compartida y después combina sus frecuencias con el resultado global mediante operaciones atómicas. Así se reduce la cantidad de actualizaciones a contadores globales, aunque sigue existiendo coordinación y contención.

Esta separación también permite mantener las temperaturas en la GPU entre el ordenamiento y el histograma. El pipeline copia los datos de entrada al dispositivo y recupera los resultados para escribirlos en CPU. Mover el arreglo al host entre ambos cálculos introduciría una transferencia adicional. Debe contabilizarse también cualquier lectura de extremos necesaria para configurar el histograma.

El enfoque híbrido puede aprovechar ambos tipos de procesador, pero no garantiza una mejora para todos los tamaños. Crear procesos, comunicar bloques, transferir memoria y lanzar kernels tiene un costo. Con pocos datos o pocas vCPU, esos costos pueden superar el ahorro. Por eso el speedup debe evaluarse con los tiempos medidos del pipeline completo, incluyendo transferencias y escritura, y no deducirse únicamente de los tiempos de los kernels.

## Aporte personal a las conclusiones

Al desarrollar mi parte, identifiqué que paralelizar requiere definir quién escribe cada dato y en qué momento puede utilizarse. En CPU conservé el orden de los bloques para reproducir exactamente las temperaturas de referencia. En GPU utilicé contadores locales por bloque, barreras y sumas atómicas para evitar actualizaciones perdidas. La corrección debe comprobarse antes del rendimiento: sumar N frecuencias no es suficiente si las muestras fueron asignadas a cubetas equivocadas.

Las pruebas de la fase CPU y del kernel de histograma fueron validadas unitariamente en `tests/tests_camila/test_camila.py` contra el oráculo secuencial, verificando la coincidencia exacta de las 100 cubetas sin discrepancias. Posteriormente, con la integración del pipeline CUDA completo y el MergeSort en la rama principal, se corroboró la aceleración en el entorno oficial de Google Colab (GPU NVIDIA T4), alcanzando un speedup de 326.36× en el histograma y 4.52× en el tiempo total para N = 1,000,000.

## Librerías empleadas en mi contribución

| Librería | Uso |
|---|---|
| `multiprocessing` | Procesos para generar temperaturas y formatear CSV por bloques. |
| `numpy` | Vectores `float32`, contadores `int32`, concatenación y comprobaciones. |
| `numba.cuda` | Kernels de inicialización y conteo, memoria compartida, barreras y atómicos. |
| `matplotlib` | Generación de `histograma.png` sin interfaz gráfica. |
| Biblioteca estándar: `pathlib`, `os`, `operator` | Rutas, número de procesos y validación de argumentos. |
| `unittest`, `tempfile`, `csv` | Pruebas reproducibles de mi código y archivos temporales. |

---

# Paso 6. Discusión técnica: aporte de Carlos

**Carlos Alburez (231311).**

## Pregunta 2. ¿Cómo CUDA permitió reducir o cambiar el tiempo de procesamiento?

La incorporación de CUDA mediante Numba transforma radicalmente el perfil de ejecución del programa, trasladando las operaciones de mayor costo asintótico ($O(N \log N)$ para MergeSort y $O(N)$ para el histograma) a los multiprocesadores de transmisión (SM) de la GPU NVIDIA T4.

### 1. Transformación del MergeSort: de recursión uniproceso a fusiones masivas en GPU
En la versión secuencial (`secuencial/histograma_seq.py`), el MergeSort recursivo fue el principal cuello de botella computacional: para $N = 10^6$ consumió **5.51 segundos**, representando más del **83.8 % del tiempo total** de ejecución. En la CPU, cada división y fusión se resolvió secuencialmente en un único hilo.

Con CUDA, el algoritmo se reformuló como un **MergeSort bottom-up iterativo por pasadas** (`paralelo/kernel_mergesort.py`):
- **Paralelismo masivo en las pasadas iniciales:** Para $w = 1$, se lanzan hasta $N/2 = 500,000$ fusiones independientes en paralelo. La GPU aprovecha sus 2,560 núcleos CUDA para resolver miles de fusiones simultáneas en una fracción de milisegundo.
- **Alternancia de buffers en VRAM (Zero-Copy Intermedio):** Entre una pasada y la siguiente, los punteros `src` y `dst` se alternan directamente en la memoria del dispositivo. Esto garantiza que las $\lceil \log_2 N \rceil \approx 20$ pasadas se ejecuten íntegramente dentro de la GPU con **cero transferencias intermedias por el bus PCIe**.
- **Comportamiento en pasadas avanzadas:** A medida que $w$ crece ($w \ge 256$), el número de fusiones independientes decrece por debajo del número de núcleos CUDA. En la última pasada ($w = N/2$), un único hilo realiza la fusión final de los dos bloques de tamaño $N/2$. A pesar de esta serialización final inherente al MergeSort, la reducción del tiempo en las primeras 15 pasadas compensa con creces, reduciendo drásticamente el tiempo frente a la CPU.

### 2. Histograma masivo con memoria compartida (Shared Memory)
El histograma secuencial requería iterar sobre el arreglo ordenado verificando los límites de las cubetas (0.23 s para $N = 10^6$). En CUDA:
- Cada bloque de 256 hilos mantiene un histograma local de 100 cubetas en **memoria compartida** (`cuda.shared.array(100, int32)`), con una latencia de acceso comparable a la caché L1 (~1 ciclo de reloj).
- Mediante un `grid-stride loop`, cada hilo clasifica su elemento con $c = \text{int}((x - \text{min}) / \text{ancho})$ y acumula en memoria compartida con `cuda.atomic.add`. La contención se aísla dentro del bloque.
- Únicamente al finalizar el bloque se realiza una reducción atómica global (`cuda.atomic.add(hist_global, ...)`). Esto reduce las operaciones atómicas globales en varios órdenes de magnitud, logrando que el histograma se complete en cuestión de milisegundos.

### 3. Impacto de las transferencias Host-Device y el tamaño de $N$
El uso de CUDA introduce dos fases inexistentes en el secuencial:
- **F2 (H2D):** Transferencia del arreglo de temperaturas sin procesar desde la RAM del host a la VRAM de la GPU (`cuda.to_device`).
- **F5 (D2H):** Recuperación del arreglo ordenado y el histograma (`copy_to_host`).

Para $N = 10^4$ (40 KB), el costo fijo de lanzamiento de kernels y la latencia del bus PCIe dominan el tiempo, haciendo que CUDA no ofrezca una ventaja notable frente a la CPU. Sin embargo, para $N = 10^6$ (4 MB), el tiempo de transferencia PCIe es despreciable comparado con los más de 5 segundos que ahorra la GPU en el cómputo. La GPU reduce el tiempo total justamente porque la complejidad de cálculo $O(N \log N)$ crece mucho más rápido que el costo de transferencia $O(N)$, alcanzando un speedup sustancial.

## Aporte personal a las conclusiones

1. **Diseño centrado en la jerarquía de memoria:** La clave para lograr un speedup efectivo en GPU no es únicamente paralelizar bucles, sino evitar el tráfico innecesario en el bus PCIe. Mantener el arreglo en VRAM entre el MergeSort y el Histograma fue determinante para no penalizar el pipeline.
2. **Trade-offs de algoritmos clásicos en SIMT:** MergeSort no es un algoritmo naturalmente amigable para arquitecturas SIMT debido a la divergencia de ramas (`if src[i] <= src[j]`) y la pérdida progresiva de paralelismo en las pasadas finales. Aun con estas limitaciones, la GPU supera a la CPU por su inmensa capacidad de cómputo en las pasadas masivas.
3. **Medición rigurosa con eventos de hardware:** El uso de `cuda.event` permitió aislar con precisión de microsegundos el tiempo real de cómputo de los kernels frente a las transferencias y el tiempo de pared del sistema operativo, garantizando una evaluación cuantitativa fidedigna.

## Librerías empleadas en mi contribución

| Librería | Uso |
|---|---|
| `numba.cuda` | Implementación del kernel `kernel_merge_pasada`, gestión de memoria en dispositivo (`device_array_like`), y temporizadores de hardware con `cuda.event`. |
| `numpy` | Manejo de tipos de datos de precisión simple (`float32`), arreglos de validación y comprobación de orden con `np.testing.assert_allclose`. |
| `psutil` / `threading` | Monitoreo concurrente en segundo plano del porcentaje de utilización de cada núcleo del CPU durante la ejecución de los benchmarks. |
| `subprocess` / `shutil` | Registro automatizado de métricas de la GPU mediante `nvidia-smi` y `nvidia-smi dmon`. |

---

# Paso 6. Discusión técnica: aporte de Marinés

**Marinés García (23391).**

## Pregunta 3. ¿Qué factores justifican o limitan la utilización de CUDA?

### Factores que justifican usar CUDA

**1. El cálculo domina el tiempo cuando N es grande.** En la versión secuencial con N = 10⁶,
el ordenamiento y el histograma suman 5.74 s de 6.57 s, es decir, el **87.4 %** del tiempo total
(Tabla 1, sección 2). MergeSort realiza casi 59 millones de operaciones elementales y crece
más rápido que N: al pasar de 10⁵ a 10⁶ su tiempo se multiplicó por 14. Mientras más grande
es N, mayor es la fracción del programa que la GPU puede acelerar.

**2. Hay mucho trabajo independiente y uniforme.** La GPU ejecuta la misma instrucción sobre
muchos datos a la vez (modelo SIMT). Las dos operaciones pesadas encajan en ese modelo:

- **Histograma:** clasificar una temperatura no depende de las demás. Cada uno de los N hilos
  calcula su cubeta con la misma fórmula, `c = (x - min) / ancho`. Las únicas dependencias son los
  100 contadores compartidos, y se resuelven con memoria compartida y sumas atómicas.
- **MergeSort bottom-up:** en cada pasada, todas las fusiones trabajan sobre segmentos que no se
  traslapan. Con N = 10⁶, la primera pasada tiene 500,000 fusiones independientes.

**3. Los datos se quedan en la GPU entre las dos operaciones.** El arreglo se copia una sola vez
al dispositivo (4 MB para N = 10⁶ en `float32`), se ordena, se calcula el histograma y solo
regresan el arreglo ordenado y 100 contadores. Las ⌈log₂N⌉ = 20 pasadas del ordenamiento no
generan transferencias intermedias. Por eso las transferencias pesan poco frente al cálculo:
**0.0046 s contra 0.2945 s** en la GPU con N = 10⁶ (las transferencias representan apenas el 1.54 % del tiempo en GPU).

**4. Diferencia de hardware.** La máquina de Colab tiene **1 núcleo físico** de CPU (2 lógicos),
mientras que la T4 tiene 2,560 núcleos CUDA. Para trabajo masivamente paralelo, la GPU tiene
mucha más capacidad de cómputo que la CPU disponible en el mismo entorno.

### Factores que limitan el uso de CUDA

**1. Ley de Amdahl: lo que se queda en la CPU pone un techo.** Si solo se acelera el cálculo
(ordenamiento + histograma) y lo demás queda igual, el speedup máximo posible es
S = 1 / (1 − p), donde p es la fracción del tiempo secuencial que ocupa el cálculo:

| N | p (cálculo / total secuencial) | Speedup máximo aunque la GPU tardara 0 s |
|---|---|---|
| 10,000 | 10.7 % | **1.12×** |
| 100,000 | 56.8 % | **2.32×** |
| 1,000,000 | 87.4 % | **7.96×** |

Con N = 10⁴, el 89 % del tiempo es escritura de archivos, así que ni una GPU infinitamente rápida
mejoraría más de 12 %. La versión paralela también acelera la escritura (con `multiprocessing`),
así que su límite real se calcula con los tiempos de CPU que efectivamente midió:
**5.70×** con N = 10⁶. El speedup total obtenido fue **4.52×**.

**2. Costos fijos que no dependen de N.** Antes de calcular nada, la GPU necesita:

- **Compilar los kernels (JIT de Numba)** y crear el contexto CUDA. Por eso se descarta la
  primera corrida como warm-up; en un programa que se ejecuta una sola vez, ese costo sí se paga.
- **Lanzar cada kernel**, lo que tiene una latencia de microsegundos. El MergeSort bottom-up
  necesita un lanzamiento por pasada (20 para N = 10⁶).
- **Copiar los datos por PCIe** en ambos sentidos.

Con N pequeño, estos costos fijos son comparables con todo el cálculo secuencial (0.037 s para
N = 10⁴), y la GPU no alcanza a compensarlos.

**3. El paralelismo del MergeSort se reduce en cada pasada.** Cada pasada duplica el tamaño de
los segmentos y reduce a la mitad el número de fusiones independientes. Con N = 10⁶:

- En las primeras pasadas hay cientos de miles de fusiones, más que suficientes para la GPU.
- A partir de segmentos de 256 elementos hay **menos fusiones que núcleos CUDA (2,560)**.
  Eso ocurre en **12 de las 20 pasadas**.
- En la última pasada, **un solo hilo** fusiona 10⁶ elementos de forma secuencial, y un núcleo
  de GPU es mucho más lento que un núcleo de CPU para trabajo secuencial.

Además, cada fusión tiene ramas que dependen de los datos (¿el siguiente elemento viene de la
mitad izquierda o de la derecha?), lo que provoca **divergencia dentro del warp**. Y cada hilo
lee sus propios segmentos, separados en memoria, por lo que los accesos **no se combinan
(coalescing)**. Por eso el speedup del ordenamiento (**18.75×**) es
menor que el del histograma (**326.36×**), que no tiene estos problemas.

**4. Contención en los contadores del histograma.** Con datos uniformes, los hilos de un bloque
reparten sus sumas entre solo 100 cubetas, así que varias sumas atómicas caen en la misma cubeta
al mismo tiempo y se serializan. La memoria compartida reduce el problema: en lugar de N sumas
atómicas globales se hacen ⌈N/256⌉ × 100 (390,700 para N = 10⁶), pero la contención dentro de
cada bloque se mantiene.

**5. Precisión numérica en la T4.** Para que los límites de cubeta coincidan exactamente con la
versión secuencial, el kernel de histograma calcula en doble precisión (`float64`). La T4 está
diseñada para `float32`: su rendimiento en `float64` es 1/32 del de `float32`. En este caso el
costo es pequeño porque el histograma hace pocas operaciones por elemento, pero en un algoritmo
con más cálculo en doble precisión sería una limitación importante.

**6. Limitaciones del entorno.** Colab gratuito asigna la GPU de forma compartida y sin
disponibilidad garantizada, y la CPU tiene un solo núcleo físico. Esto también limita la fase de
CPU: `multiprocessing` solo puede usar 2 hilos sobre el mismo núcleo, así que la preparación y la
escritura casi no se aceleran. La preparación incluso puede volverse más lenta por el costo de
crear procesos (**0.14×** con N = 10⁶).

**7. Complejidad de desarrollo.** El código CUDA exige pensar en hilos, bloques, barreras y
condiciones de carrera. La recursión del MergeSort original no se traslada bien a la GPU (cada
hilo tiene una pila muy pequeña y una llamada recursiva no genera paralelismo), así que el
algoritmo tuvo que reescribirse de forma iterativa, por pasadas. Además, los errores no siempre son visibles, porque un
histograma puede sumar N y aun así tener elementos en la cubeta equivocada. Por eso cada kernel
se validó contra la versión secuencial antes de medir tiempos.

### Conclusión

CUDA **se justifica cuando N es grande y el cálculo domina el tiempo total**: con N = 10⁶ el
cálculo es el 87.4 % del tiempo secuencial y la GPU lo reduce de 5.74 s a **0.2992 s** (19.19× de speedup en cómputo + transferencias). **No se justifica con N pequeño**: con N = 10⁴ el cuello de botella es la
escritura de archivos, que se queda en la CPU, y el speedup máximo posible es apenas 1.12×. En
general, el beneficio de CUDA depende de qué fracción del programa es paralelizable (Amdahl), de
cuánto cuesta mover los datos y de qué tan bien se adapta el algoritmo al modelo SIMT de la GPU.


