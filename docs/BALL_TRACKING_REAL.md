# Seguimiento de bola en video real — primera iteración

## Arquitectura

```text
frames anterior/actual/siguiente + fondo mediano + ROI de escena
                    │
                    ▼
 ball_detection.py: blobs de movimiento/foreground/contraste/color
                    │  0..N BallCandidate
                    ▼
 ball_video_tracking.py: bootstrap temporal (>=3 observaciones)
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
 velocidad constante       BallState + propagación física
 (comparador)               + associate_observations()
                               │
                  OBSERVED / PREDICTED / LOST
                               │
           mesa + muñeca derecha + discontinuidades
                               ▼
                 eventos y dataset sincronizado
```

El detector conserva centro, caja, área, tamaño, circularidad, elongación,
vector de blur, intensidad, saturación, contraste, movimiento, foreground y la
contribución de cada evidencia a la confianza. Una observación visual no se
convierte en una predicción ni viceversa.

El tracker reutiliza sin cambios `BallState`, `propagate_state()`,
`associate_observations()` y los parámetros de `ball_tracking.py`. La física se
aplica dentro de tramos de vuelo libre. Una pérdida junto a la muñeca derecha
cierra el tramo como posible impacto; no se propaga el mismo estado a través de
la colisión.

## Geometría y unidades

`config/ball_tracking_scenes.json` contiene ROI y cuatro esquinas aproximadas de
mesa que pueden editarse manualmente por clip. El largo reglamentario de 2.74 m
se usa como prior horizontal y existe un prior vertical explícito. Esto solo
define una proyección planar útil para gating. No calibra profundidad, óptica ni
la posición 3D de cámara; por tanto, la salida defendible es velocidad aparente
en px/s. Los estados métricos internos son auxiliares y no se publican como
velocidad real.

Los tiempos provienen de `ffprobe best_effort_timestamp_time`. El fallback
`frame/fps` solo se activa si esos timestamps no pueden obtenerse y queda
registrado en `summary.json`.

## Prueba actual

- Fácil: `20251212_132025_1.mp4`.
- Difícil: `20251212_140101_1.mp4`, con más blur y oclusión.
- El video largo no se procesa.
- Overlay, CSV por frame, eventos y dataset de golpes permanecen locales bajo
  `data/processed/ball_tracking_real/`.
- `summary.json` y `REPORT.md` condensan resultados y limitaciones.

En el clip fácil se anotaron 25 puntos en dos tramos. Los centroides del
detector se usaron para iniciar la anotación y después se aceptaron/corrigieron
visualmente sobre el original; esta referencia es útil para identidad y falsos
positivos, pero hace optimista el error de localización.

El tramo principal auditado abarca los frames 82–111. La bola llega desde la
mesa, bota aproximadamente en el frame 89, se aproxima a la raqueta en el 96,
queda sin track aceptado en 97–99 y se reinicializa sobre el vuelo saliente en
100. Otro tramo contiene una predicción en 214 y recuperación observada en 215.

## Interpretación

- `OBSERVED`: candidato visual aceptado por asociación.
- `PREDICTED`: propagación sin candidato compatible; naranja en el overlay.
- `LOST`: se agotó la ventana de cuatro predicciones o se cerró un tramo por
  posible impacto.
- `UNINITIALIZED`: aún no hay tres detecciones temporalmente coherentes.

La cobertura global incluye regiones no anotadas y no equivale a precisión. El
informe separa cobertura de la pequeña auditoría manual. Ejemplos de falsos
positivos del detector aislado se registran por frame; las pérdidas y
reinicializaciones se conservan, no se rellenan como si fueran observaciones.

Los eventos se llaman `possible_racket_contact` y `possible_table_bounce`.
El dataset por golpe sincroniza timestamp, velocidad aparente de bola, dirección
de imagen, frames observados/predichos, velocidad de muñeca derecha, ángulo de
codo, velocidad angular de codo y confianza. No se realiza estadística compleja.

## Spin

Magnus continúa desactivado en el tracker operativo. Como diagnóstico se comparan
gravedad, gravedad con drag y una rejilla fija de Magnus de ±200 rad/s sobre
pequeños tramos anotados, ajustando tres puntos y reteniendo los restantes. Las
variantes Magnus pueden reducir el error en esa rejilla, pero una sola vista a
~30 fps, la proyección planar aproximada, el motion blur y tan pocos puntos no
permiten separar spin de error geométrico ni demostrar estabilidad. El resultado
correcto de esta fase sigue siendo `spin_not_identifiable`; no se reporta un
vector de spin medido.

## Ejecución

```powershell
.venv\Scripts\python.exe scripts/process_ball_tracking_real.py
.venv\Scripts\python.exe -m pytest tests/test_ball_tracking.py tests/test_ball_detection.py tests/test_ball_video_tracking.py -q
```

El MP4 puede omitirse con `--no-overlay`. Activarlo no modifica la detección: el
dibujo opera sobre una copia del frame. Dos ejecuciones sin overlay del clip
fácil produjeron el mismo SHA-256 para `frames.csv`.

## Límites pendientes

La cámara registra cerca de 30 fps y la bola recorre decenas de píxeles durante
una exposición; centro y velocidad tienen incertidumbre temporal/espacial. La
mesa se marcó de forma aproximada. El contacto con la raqueta del oponente no
usa todavía pose del oponente y puede perderse. Movimientos blancos/contrastados
de personas y fondo siguen generando candidatos; solo la asociación los reduce.
Hace falta ampliar anotación independiente, calibrar cámara/mesa y adquirir más
fps antes de informar m/s, drag ajustado o spin efectivo.
