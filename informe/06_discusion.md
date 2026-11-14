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


