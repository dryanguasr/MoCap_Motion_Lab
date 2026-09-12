# Validación de clips y recuperación manual — 2026-09-09

Implementación comprobada con **67 pruebas aprobadas**. Se conservan las pruebas
del flujo automático y se añaden casos de edición de intervalos, protección de
geometría, semillas visibles/ocluídas, clic erróneo, persistencia/reanudación,
conflicto con ancla clásica, controles de inspección y exportaciones VFR.

## Preparación del video

- Fuente: `Ma-Long-and-Fan-Zhendong-Training-T2-Diamond-2019-Malaysia.mp4`.
- 12 552 fotogramas con PTS reales, resolución 2156 × 1214.
- **20 propuestas de intercambio y 6 propuestas de toma**, todavía sin aceptación
  del usuario. No equivalen a 20 clips definitivos ni a seis tomas confirmadas.
- La caché histórica RacketVision usaba tiempo por fps medio: error máximo de
  0,616948 s respecto a PTS. Se reutiliza por índice original y hash, y se corrigen
  los timestamps de las observaciones sin modificar la caché.
- Se detectó que el salto de OpenCV por índice devuelve imágenes desfasadas en
  esta fuente VFR. La revisión y el seguimiento usan búsquedas por PTS con FFmpeg;
  se contrastó el fotograma 3000 con decodificación secuencial y se incorporó una
  prueba VFR con imágenes distinguibles para todos los índices.

## Pruebas sobre imágenes reales

Tres extractos diagnósticos de 61 fotogramas, con semillas aproximadas fijadas por
el agente sobre imágenes originales. Son pruebas del mecanismo, **no tres
intercambios completos aceptados por el usuario**.

| Extracto, índices originales inclusivos | Tipo de semilla | Recuperación inicial | Inspección visual inicial |
|---|---|---|---|
| 1500–1560 | Estimada, cerca de la malla | 1502 | Bola/estela en 1502–1505 |
| 3000–3060 | Visible, blur intenso | 3003 | Bola/estela en 3003–3006 |
| 6000–6060 | Visible aproximada, cerca de mesa | 6002 | Bola/estela en 6002 y 6004 |

En los tres casos se verificaron la correspondencia temporal de MP4 y la igualdad
de resultados al repetir las mismas correcciones. Las hojas de contacto y videos
están en `data/processed/diagnostics/manual_recovery_smoke/`; el registro detallado
es `report.json` en esa carpeta.

Persisten pérdidas posteriores, especialmente cerca de manos/raquetas. No se
deduce precisión global ni cobertura fiable de estos ejemplos. El modo interactivo
se detiene en esas pérdidas para pedir otra intervención; el diagnóstico dejó
continuar el motor para mostrar las limitaciones.

## Pendiente de la sesión con el usuario

Revisar límites, separar o unir propuestas cuando corresponda, comprobar mesa y
malla al inicio/centro/final de cada toma y aceptar los intercambios. Después,
comprobar cada recuperación durante el seguimiento interactivo. Las exportaciones
definitivas se habilitan para los intervalos aceptados con geometría revisada.

Entradas: `review_professionals.cmd` y `track_professionals.cmd`.
Controles y comandos: `docs/RALLY_MANUAL_RECOVERY.md`.
