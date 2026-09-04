# Primera prueba real de seguimiento de bola

Resultados exploratorios. Observación visual, predicción y pérdida se conservan como estados distintos. La calibración es planar y aproximada: no se reporta velocidad métrica ni spin medido.

| Video | Dificultad | Físico base | Multimodal | Perdidos base→fusión | Contactos / botes |
|---|---|---:|---:|---:|---:|
| Ma-Long-and-Fan-Zhendong-Training-T2-Diamond-2019-Malaysia.mp4 | long_professional_session_fixed_camera_fast_ball_and_occlusions | 0.1% | 43.1% | 11470→7134 | 44 / 25 |

Los archivos `metrics/*__ball_tracking__frames.csv` y los MP4 de `videos/` permanecen locales. Los conteos no prueban exactitud: la auditoría manual es pequeña y debe ampliarse antes de ajustar umbrales o estudiar Magnus.

Variables identificables por ahora: coordenadas 2D, estado observado/predicho, cobertura y velocidad aparente px/s si hay suficientes observaciones. No identificables: posición/velocidad 3D calibradas y spin; resultado de spin: `spin_not_identifiable`.

Las detecciones de eventos continúan siendo hipótesis. Las anotaciones actuales no son exhaustivas, por lo que permiten medir recuperación alrededor de eventos conocidos, pero no precisión global de eventos ni PCK de raqueta.