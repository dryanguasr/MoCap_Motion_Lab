# Primera prueba real de seguimiento de bola

Resultados exploratorios. Observación visual, predicción y pérdida se conservan como estados distintos. La calibración es planar y aproximada: no se reporta velocidad métrica ni spin medido.

| Video | Dificultad | Candidatos aislados | Temporal | Físico proyectado | Observados / predichos | Contactos / botes |
|---|---|---:|---:|---:|---:|---:|
| 20251212_132025_1.mp4 | easy_candidate | 99.3% | 80.7% | 80.7% | 201 / 33 | 7 / 9 |
| 20251212_140101_1.mp4 | motion_blur_and_occlusion | 99.4% | 77.7% | 74.1% | 189 / 40 | 7 / 7 |

## Auditoría manual: 20251212_132025_1.mp4

| Método | Cobertura | Mediana / P95 / máximo / RMSE (px) | Falsos positivos observables | Frames perdidos |
|---|---:|---:|---:|---:|
| Detector aislado | 100.0% | 0.0 / 825.9 / 966.5 / 282.1 | [97, 98, 108, 215] | [] |
| Continuidad temporal | 92.0% | 162.8 / 1378.9 / 1491.2 / 679.8 | [98, 99, 100, 102, 104, 106, 208, 210, 212, 213, 214, 215] | [108, 109] |
| Tracker físico proyectado | 88.0% | 0.0 / 0.0 / 37.2 / 7.9 | [] | [97, 98, 99] |

Referencia escasa (25 puntos) y no independiente para precisión subpíxel: los centroides se aceptaron/corrigieron visualmente. El tracker físico tuvo 1 predicción(es) anotada(s), con mediana de error 37.2 px. No extrapolar estas cifras al video completo.

### Comparación exploratoria de vuelo

| Segmento | Frames de ajuste | Frames retenidos | Gravedad | Gravedad + drag | Magnus -200 | Magnus +200 | Mejor en la rejilla |
|---|---:|---:|---:|---:|---:|---:|---|
| incoming_before_bounce | [82, 84, 86] | [88] | 5.8 | 3.0 | 6.6 | 2.3 | magnus_plus_200_rad_s |
| after_bounce_before_main_contact | [90, 92, 94] | [96] | 3.1 | 3.0 | 1.3 | 4.6 | magnus_minus_200_rad_s |
| outgoing_after_main_contact | [100, 102, 104] | [106] | 24.2 | 16.1 | 22.5 | 9.8 | magnus_plus_200_rad_s |
| toward_opponent | [208, 210, 212] | [213] | 8.1 | 5.1 | 7.1 | 3.7 | magnus_plus_200_rad_s |

Resultado: `spin_not_identifiable`. Que una variante Magnus gane esta rejilla pequeña no identifica spin: la proyección es monocular aproximada, hay pocos puntos retenidos y el parámetro no se estimó ni se validó de forma independiente.

## Auditoría manual: 20251212_140101_1.mp4

| Método | Cobertura | Mediana / P95 / máximo / RMSE (px) | Falsos positivos observables | Frames perdidos |
|---|---:|---:|---:|---:|
| Detector aislado | 100.0% | 0.0 / 202.6 / 368.4 / 116.5 | [186] | [] |
| Continuidad temporal | 100.0% | 692.4 / 944.8 / 975.3 / 687.4 | [177, 178, 179, 180, 181, 182, 183, 184, 185, 186] | [] |
| Tracker físico proyectado | 100.0% | 0.0 / 0.0 / 0.0 / 0.0 | [] | [] |

Referencia escasa (10 puntos) y no independiente para precisión subpíxel: los centroides se aceptaron/corrigieron visualmente. No extrapolar estas cifras al video completo.

### Comparación exploratoria de vuelo

| Segmento | Frames de ajuste | Frames retenidos | Gravedad | Gravedad + drag | Magnus -200 | Magnus +200 | Mejor en la rejilla |
|---|---:|---:|---:|---:|---:|---:|---|
| blurred_flight | [177, 178, 179] | [180, 181, 182, 183, 184, 185] | 61.4 | 38.2 | 50.1 | 30.7 | magnus_plus_200_rad_s |

Resultado: `spin_not_identifiable`. Que una variante Magnus gane esta rejilla pequeña no identifica spin: la proyección es monocular aproximada, hay pocos puntos retenidos y el parámetro no se estimó ni se validó de forma independiente.

`frames.csv` y el MP4 diagnóstico permanecen locales. Los conteos no prueban exactitud: la auditoría manual es pequeña y debe ampliarse antes de ajustar umbrales o estudiar Magnus.

Variables identificables por ahora: coordenadas 2D, estado observado/predicho, cobertura y velocidad aparente px/s si hay suficientes observaciones. No identificables: posición/velocidad 3D calibradas y spin; resultado de spin: `spin_not_identifiable`.