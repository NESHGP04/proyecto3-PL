# Paso 6. Discusión técnica: aporte de Camila

**Camila Richter (23183).** 

## Pregunta 1. ¿Por qué conviene procesar archivos en CPU multinúcleo y delegar solo la matemática a la GPU?

La CPU es adecuada para coordinar procesos, generar bloques de entrada, convertir números a texto y acceder al sistema de archivos. En nuestra propuesta, `multiprocessing.Pool` reparte la generación de temperaturas y el formateo del CSV entre los núcleos disponibles. Cada proceso trabaja con un bloque independiente y el proceso principal escribe los textos en orden, evitando que se mezclen las filas del archivo.

La GPU se aprovecha cuando muchos datos requieren operaciones similares. Por eso recibe el ordenamiento con MergeSort y el conteo de las 100 cubetas. Para el histograma, cada bloque cuenta primero en memoria compartida y después combina sus frecuencias con el resultado global mediante operaciones atómicas. Así se reduce la cantidad de actualizaciones a contadores globales, aunque sigue existiendo coordinación y contención.

Esta separación también permite mantener las temperaturas en la GPU entre el ordenamiento y el histograma. El pipeline copia los datos de entrada al dispositivo y recupera los resultados para escribirlos en CPU. Mover el arreglo al host entre ambos cálculos introduciría una transferencia adicional. Debe contabilizarse también cualquier lectura de extremos necesaria para configurar el histograma.

El enfoque híbrido puede aprovechar ambos tipos de procesador, pero no garantiza una mejora para todos los tamaños. Crear procesos, comunicar bloques, transferir memoria y lanzar kernels tiene un costo. Con pocos datos o pocas vCPU, esos costos pueden superar el ahorro. Por eso el speedup debe evaluarse con los tiempos medidos del pipeline completo, incluyendo transferencias y escritura, y no deducirse únicamente de los tiempos de los kernels.

## Aporte personal a las conclusiones

Al desarrollar mi parte, identifiqué que paralelizar requiere definir quién escribe cada dato y en qué momento puede utilizarse. En CPU conservé el orden de los bloques para reproducir exactamente las temperaturas de referencia. En GPU utilicé contadores locales por bloque, barreras y sumas atómicas para evitar actualizaciones perdidas. La corrección debe comprobarse antes del rendimiento: sumar N frecuencias no es suficiente si las muestras fueron asignadas a cubetas equivocadas.

Las pruebas de CPU ejecutadas y el estado de las pruebas CUDA se registran en `pruebas_camila/VALIDACION.md`. Todavía no se pueden afirmar mejoras de rendimiento ni coincidencia con el histograma secuencial del equipo, porque esa función y la ejecución del pipeline completo permanecen pendientes en el avance recibido.

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

# Paso 6. Discusión técnica: aporte de Marinés

**Marinés García (23391).**

> ⚠️ Los valores marcados con ⟨…⟩ salen de `resultados/tabla_comparativa.md`, que se genera al
> correr `--modo graficas` con los tiempos de la versión CUDA. El resto de los números sale de la
> medición secuencial en Colab (sección 2) y del conteo de operaciones (sección 1).

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
⟨H→D + D→H⟩ s contra ⟨ordenamiento + histograma⟩ s en la GPU con N = 10⁶.

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
⟨límite de Amdahl de la tabla comparativa⟩ con N = 10⁶. El speedup obtenido fue ⟨speedup total⟩.

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
(coalescing)**. Por eso el speedup del ordenamiento (⟨speedup del kernel de ordenamiento⟩) es
menor que el del histograma (⟨speedup del kernel de histograma⟩), que no tiene estos problemas.
*(Este análisis corresponde al diseño bottom-up del PLAN; si la implementación de Carlos cambia,
hay que revisarlo.)*

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
crear procesos (⟨speedup de preparación⟩).

**7. Complejidad de desarrollo.** El código CUDA exige pensar en hilos, bloques, barreras y
condiciones de carrera. La recursión del MergeSort original no se traslada bien a la GPU (cada
hilo tiene una pila muy pequeña y una llamada recursiva no genera paralelismo), así que el
algoritmo tuvo que reescribirse de forma iterativa, por pasadas. Además, los errores no siempre son visibles, porque un
histograma puede sumar N y aun así tener elementos en la cubeta equivocada. Por eso cada kernel
se validó contra la versión secuencial antes de medir tiempos.

### Conclusión

CUDA **se justifica cuando N es grande y el cálculo domina el tiempo total**: con N = 10⁶ el
cálculo es el 87.4 % del tiempo secuencial y la GPU lo reduce de 5.74 s a ⟨cálculo GPU +
transferencias⟩ s. **No se justifica con N pequeño**: con N = 10⁴ el cuello de botella es la
escritura de archivos, que se queda en la CPU, y el speedup máximo posible es apenas 1.12×. En
general, el beneficio de CUDA depende de qué fracción del programa es paralelizable (Amdahl), de
cuánto cuesta mover los datos y de qué tan bien se adapta el algoritmo al modelo SIMT de la GPU.
---

## Pregunta 3. ¿Qué factores justifican o limitan la utilización de CUDA?

**Marinés García (23391).**

> ⚠️ Los valores marcados con ⟨…⟩ se llenan con `resultados/tabla_comparativa.md` cuando estén los tiempos CUDA oficiales.

### Factores que justifican usar CUDA

1. **Hay un cuello de botella claro y paralelizable.** En la versión secuencial, con N = 10⁶, el ordenamiento y el histograma ocupan el **87.4 %** del tiempo total (5.51 s + 0.23 s de 6.57 s). Según el Paso 1, MergeSort hace casi 59 millones de operaciones elementales. Es justo el tipo de trabajo que conviene llevar a la GPU: muchas operaciones iguales sobre datos independientes.
2. **El trabajo se puede dividir en piezas independientes.** Clasificar cada temperatura en su cubeta no depende de las demás, y las fusiones de una misma pasada de MergeSort trabajan sobre rangos que no se traslapan. Miles de hilos pueden trabajar a la vez sin coordinarse más allá de las sumas atómicas y las barreras entre pasadas.
3. **El trabajo crece más rápido que los datos.** MergeSort es O(N·log₂N): al pasar de 10⁵ a 10⁶ elementos, su tiempo secuencial creció ×14.0, mientras que los datos a transferir solo crecen ×10. Mientras más grande es N, más cálculo se hace por cada byte que viaja a la GPU, y mejor se amortiza la transferencia. Por eso se espera que el speedup aumente con N (⟨speedup total con N = 10⁴, 10⁵ y 10⁶⟩).
4. **Los datos se quedan en la GPU entre fases.** El arreglo se copia una sola vez a la GPU, se ordena y se cuenta ahí mismo, y solo regresa al final. Las transferencias quedan en ⟨h2d + d2h⟩ s con N = 10⁶: apenas ⟨x⟩ % del total.

### Factores que limitan la utilización de CUDA

1. **Ley de Amdahl: lo que se queda en la CPU pone un techo.** Aunque los kernels fueran infinitamente rápidos, el pipeline sigue teniendo que preparar los datos y escribir los archivos en la CPU. Con los tiempos medidos de la versión paralela, el speedup máximo posible con N = 10⁶ es **⟨límite de Amdahl⟩×**, aunque la GPU tardara 0 s. Con N = 10⁴ la situación es más extrema: en el secuencial, la escritura ya es el **89 %** del total. Ahí la GPU casi no tiene qué acelerar y el speedup total queda cerca de 1×.
2. **Costos fijos que no dependen de N.** Compilar los kernels con Numba la primera vez (warm-up), crear el contexto de la GPU, lanzar cada kernel y crear los procesos del `Pool` cuesta lo mismo con 10⁴ que con 10⁶ datos. Con N pequeño esos costos pesan más que el cálculo: la preparación paralela tarda ⟨preparación CUDA⟩ s, contra solo 0.001 s en la versión secuencial, porque crear procesos cuesta más que generar 10⁴ números.
3. **Transferencias por PCIe.** Copiar datos entre la RAM y la memoria de la GPU tiene latencia y ancho de banda limitados. Con pocos datos, el tiempo de mover el arreglo puede ser comparable al del cálculo mismo.
4. **MergeSort no es igual de paralelo en todas sus pasadas.** En la primera pasada hay N/2 fusiones independientes, pero cada pasada reduce a la mitad el número de hilos activos. En la última, **un solo hilo** fusiona los N elementos mientras el resto de la GPU espera. Además, las log₂N ≈ 20 pasadas deben ir en orden. Por eso el speedup del ordenamiento (⟨×⟩) es mucho menor que el del histograma (⟨×⟩), que sí usa todos los hilos todo el tiempo.
5. **Hardware de Colab.** La máquina tiene 1 núcleo físico con 2 hilos. La fase de CPU con `multiprocessing` casi no puede ganar: dos procesos compiten por el mismo núcleo, y la escritura (la fase que más pesa después del cálculo) solo mejora ⟨×⟩. Además, la GPU T4 se comparte en un entorno virtual, lo que agrega variación entre corridas.
6. **Contención en los atómicos.** El histograma usa sumas atómicas. Si muchas temperaturas cayeran en pocas cubetas, los hilos esperarían su turno para la misma posición. Con nuestros datos uniformes (~10,000 por cubeta) la contención es baja, pero con datos concentrados el kernel sería más lento.
7. **Complejidad y depuración.** El código CUDA requiere pensar en hilos, bloques, memoria compartida, barreras y condiciones de carrera, y validar los resultados contra la versión secuencial antes de confiar en los tiempos. Para un problema pequeño o que se ejecuta una sola vez, ese costo de desarrollo puede no justificarse.

### Conclusión

CUDA se justifica cuando el problema tiene **mucho cálculo repetitivo e independiente** y **N es grande**: ahí la GPU reduce el cálculo de ⟨5.74⟩ s a ⟨…⟩ s. Su beneficio sobre el programa completo está limitado por todo lo que no se acelera (archivos, procesos y transferencias). Por eso el speedup total (⟨…⟩×) es mucho menor que el de los kernels, y nunca puede superar el límite de Amdahl (⟨…⟩×). Para N pequeño, la versión paralela no aporta una mejora significativa.


