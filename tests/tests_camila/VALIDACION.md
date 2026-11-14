# Validación de la contribución de Camila

Fecha: 14 de noviembre de 2026.

## Ejecución en Google Colab

Desde la raíz del proyecto se ejecutó:

python -m tests.tests_camila.test_camila

Las pruebas usan multiprocessing en modo spawn.

Resultado: 10 pruebas ejecutadas, 9 aprobadas y 1 omitida.
Duración de la suite: 11.695 segundos. Este tiempo no es un benchmark
del algoritmo ni permite calcular speedup.

## Resultados

| Comprobación | Resultado |
|---|---|
| Generación idéntica a generar_secuencial para N = 1, 7, 257 y 10 000, con 1 y 2 procesos | Aprobada |
| Preparación de un vector vacío float32 | Aprobada |
| Rechazo de argumentos inválidos | Aprobada |
| CSV de 1 003 temperaturas: encabezado, índices, valores y orden | Aprobada |
| Archivos de texto idénticos con 1 y 2 procesos | Aprobada |
| Archivo DAT con 100 cubetas y generación de un PNG válido | Aprobada |
| Escritura para datos vacíos y constantes | Aprobada |
| Rechazo de un histograma con cuentas inconsistentes | Aprobada |
| Histograma CUDA con datos vacíos, constantes, extremos y fronteras | Aprobada |
| Histograma CUDA con N = 1 025 y reinicialización entre llamadas | Aprobada |
| Grid-stride con 2 bloques de 32 hilos y N = 2 049 | Aprobada |
| Comparación contra el histograma secuencial del equipo | Omitida: la función secuencial todavía está pendiente |

La tabla desglosa comprobaciones realizadas dentro de las 10 pruebas;
cada fila no representa necesariamente una prueba independiente.

Las tres pruebas CUDA aprobadas comparan las frecuencias contra una
referencia independiente basada en np.searchsorted y comprueban que
la suma de las frecuencias sea igual a N.

## Avisos de rendimiento

Durante las pruebas aparecieron mensajes NumbaPerformanceWarning
por lanzar grids pequeños.

Estos avisos no causaron fallos. Las pruebas utilizan pocos datos para
verificar casos específicos y no buscan aprovechar toda la GPU.

## Validación pendiente

Cuando Marinés complete secuencial.histograma_seq.histograma,
se debe repetir:

python -m tests.tests_camila.test_camila

La comparación actualmente omitida deberá aprobarse antes de realizar
los benchmarks definitivos del equipo.

## Límite del alcance

Estas pruebas validan la preparación de datos, la escritura de archivos
y el kernel de histograma de Camila.

No validan el kernel MergeSort, el pipeline completo ni su rendimiento.
La comparación del ordenamiento GPU contra np.sort y los benchmarks
completos corresponden a la integración de Carlos.

Los archivos compartidos y los archivos asignados a Marinés y Carlos
se conservaron sin cambios.
