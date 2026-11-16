**N = 10,000**

| Fase | Secuencial (s) | CUDA (s) | Speedup |
|---|---|---|---|
| Preparación (CPU) | 0.0010 | 0.0342 | 0.03× |
| Ordenamiento (kernel) | 0.0344 | 0.0083 | 4.16× |
| Histograma (kernel) | 0.0023 | 0.0004 | 5.75× |
| Cálculo GPU + transferencias | 0.0367 | 0.0095 | 3.86× |
| Escritura (CPU) | 0.3059 | 0.3366 | 0.91× |
| **Total** | 0.3436 | 0.3810 | **0.90×** |
| *Límite de Amdahl (GPU en 0 s)* | 0.3436 | 0.3708 | *0.93×* |

**N = 100,000**

| Fase | Secuencial (s) | CUDA (s) | Speedup |
|---|---|---|---|
| Preparación (CPU) | 0.0016 | 0.0304 | 0.05× |
| Ordenamiento (kernel) | 0.3942 | 0.0646 | 6.10× |
| Histograma (kernel) | 0.0208 | 0.0004 | 49.34× |
| Cálculo GPU + transferencias | 0.4150 | 0.0660 | 6.29× |
| Escritura (CPU) | 0.3137 | 0.3238 | 0.97× |
| **Total** | 0.7303 | 0.4208 | **1.74×** |
| *Límite de Amdahl (GPU en 0 s)* | 0.7303 | 0.3542 | *2.06×* |

**N = 1,000,000**

| Fase | Secuencial (s) | CUDA (s) | Speedup |
|---|---|---|---|
| Preparación (CPU) | 0.0099 | 0.0631 | 0.16× |
| Ordenamiento (kernel) | 5.5106 | 0.2899 | 19.01× |
| Histograma (kernel) | 0.2313 | 0.0006 | 376.49× |
| Cálculo GPU + transferencias | 5.7420 | 0.2949 | 19.47× |
| Escritura (CPU) | 0.8153 | 0.9911 | 0.82× |
| **Total** | 6.5672 | 1.3499 | **4.86×** |
| *Límite de Amdahl (GPU en 0 s)* | 6.5672 | 1.0543 | *6.23×* |

**Desglose de la versión CUDA (promedio ± desviación, s)**

| N | Preparación | Transferencia H→D | Ordenamiento | Histograma | Transferencia D→H | Escritura | Total |
|---|---|---|---|---|---|---|---|
| 10,000 | 0.0342 ± 0.0089 | 0.0007 ± 0.0001 | 0.0083 ± 0.0002 | 0.0004 ± 0.0001 | 0.0002 ± 0.0000 | 0.3366 ± 0.0696 | 0.3810 ± 0.0756 |
| 100,000 | 0.0304 ± 0.0022 | 0.0007 ± 0.0001 | 0.0646 ± 0.0001 | 0.0004 ± 0.0001 | 0.0003 ± 0.0001 | 0.3238 ± 0.0100 | 0.4208 ± 0.0101 |
| 1,000,000 | 0.0631 ± 0.0121 | 0.0019 ± 0.0004 | 0.2899 ± 0.0366 | 0.0006 ± 0.0001 | 0.0024 ± 0.0006 | 0.9911 ± 0.2149 | 1.3499 ± 0.2217 |
