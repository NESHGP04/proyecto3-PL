**N = 10,000**

| Fase | Secuencial (s) | CUDA (s) | Speedup |
|---|---|---|---|
| Preparación (CPU) | 0.0010 | 0.0452 | 0.02× |
| Ordenamiento (kernel) | 0.0344 | 0.0085 | 4.04× |
| Histograma (kernel) | 0.0023 | 0.0008 | 2.85× |
| Cálculo GPU + transferencias | 0.0367 | 0.0103 | 3.55× |
| Escritura (CPU) | 0.3059 | 0.4332 | 0.71× |
| **Total** | 0.3436 | 0.4903 | **0.70×** |
| *Límite de Amdahl (GPU en 0 s)* | 0.3436 | 0.4783 | *0.72×* |

**N = 100,000**

| Fase | Secuencial (s) | CUDA (s) | Speedup |
|---|---|---|---|
| Preparación (CPU) | 0.0016 | 0.0376 | 0.04× |
| Ordenamiento (kernel) | 0.3942 | 0.0648 | 6.08× |
| Histograma (kernel) | 0.0208 | 0.0005 | 42.66× |
| Cálculo GPU + transferencias | 0.4150 | 0.0665 | 6.24× |
| Escritura (CPU) | 0.3137 | 0.3862 | 0.81× |
| **Total** | 0.7303 | 0.4911 | **1.49×** |
| *Límite de Amdahl (GPU en 0 s)* | 0.7303 | 0.4238 | *1.72×* |

**N = 1,000,000**

| Fase | Secuencial (s) | CUDA (s) | Speedup |
|---|---|---|---|
| Preparación (CPU) | 0.0099 | 0.0700 | 0.14× |
| Ordenamiento (kernel) | 5.5106 | 0.2938 | 18.75× |
| Histograma (kernel) | 0.2313 | 0.0007 | 326.36× |
| Cálculo GPU + transferencias | 5.7420 | 0.2992 | 19.19× |
| Escritura (CPU) | 0.8153 | 1.0832 | 0.75× |
| **Total** | 6.5672 | 1.4532 | **4.52×** |
| *Límite de Amdahl (GPU en 0 s)* | 6.5672 | 1.1531 | *5.70×* |

**Desglose de la versión CUDA (promedio ± desviación, s)**

| N | Preparación | Transferencia H→D | Ordenamiento | Histograma | Transferencia D→H | Escritura | Total |
|---|---|---|---|---|---|---|---|
| 10,000 | 0.0452 ± 0.0290 | 0.0007 ± 0.0001 | 0.0085 ± 0.0012 | 0.0008 ± 0.0009 | 0.0003 ± 0.0004 | 0.4332 ± 0.3595 | 0.4903 ± 0.3923 |
| 100,000 | 0.0376 ± 0.0070 | 0.0008 ± 0.0001 | 0.0648 ± 0.0005 | 0.0005 ± 0.0001 | 0.0004 ± 0.0000 | 0.3862 ± 0.0532 | 0.4911 ± 0.0605 |
| 1,000,000 | 0.0700 ± 0.0139 | 0.0020 ± 0.0004 | 0.2938 ± 0.0490 | 0.0007 ± 0.0001 | 0.0026 ± 0.0007 | 1.0832 ± 0.1805 | 1.4532 ± 0.1863 |
