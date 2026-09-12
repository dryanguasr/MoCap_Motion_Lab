# Seguimiento de bola en video real — primera iteración

## Arquitectura

```text
 detector clásico ───────────────┐
 RacketVision BallTrack ─────────┼─ fusión de evidencia ─ tracker físico
 pose corporal izquierda/derecha ┤                         │
 RacketPose (5 puntos por pala) ──┘                         ▼
                                      FREE_FLIGHT / BOUNCE_PENDING /
                                      CONTACT_PENDING / POST_EVENT
                                                         │
                                  suavizado offline de 9 frames
                                                         ▼
                                  OBSERVED / PREDICTED / LOST
```

El detector conserva centro, caja, área, tamaño, circularidad, elongación,
vector de blur, intensidad, saturación, contraste, movimiento, foreground y la
contribución de cada evidencia a la confianza. Una observación visual no se
convierte en una predicción ni viceversa.

El detector clásico conserva la autoridad cuando tiene una asociación válida.
Los heatmaps aprendidos se usan como evidencia adicional y, especialmente, para
recuperar huecos; así no desplazan una observación fiable durante vuelo normal.
El cuerpo penaliza distractores salvo dentro de la zona de interacción. La pala
se asocia globalmente al jugador mediante mango–muñeca, lado, bbox y continuidad.
Cuando el modelo no la encuentra se usa una cápsula muñeca–antebrazo marcada
explícitamente como predicción de baja confianza.

Un rebote solo abre su ventana cerca del borde superior de la mesa y con
movimiento entrante. El contacto usa distancia bola–cabeza, velocidad de cierre
y tiempo a máxima proximidad. En estas ventanas la compuerta se ensancha y se
conservan tres hipótesis. Los huecos de hasta dos frames se reconstruyen con una
rama cuadrática de rebote o interpolación de evento, siempre como `PREDICTED`.

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
registrado en cada archivo `summaries/*__ball_tracking__summary.json`.

## Prueba actual

- Fácil: `20251212_132025_1.mp4`.
- Difícil: `20251212_140101_1.mp4`, con más blur y oclusión.
- Para videos largos, detectar primero cambios de plano, zoom o reencuadre. Dividir el material en clips de toma fija antes del análisis: la geometría de mesa y malla se anota y valida por clip, y nunca se propaga entre planos distintos. Solo se agregan métricas entre clips después de conservar su procedencia temporal y geométrica.
- Los overlays quedan en `data/processed/videos/`, los CSV por frame en
  `metrics/`, los eventos en `events/` y los golpes en `datasets/`.
- Los resúmenes quedan en `summaries/` y el informe condensado en
  `reports/batch__ball_tracking__report.md`.

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
powershell -ExecutionPolicy Bypass -File scripts/setup_racketvision.ps1
.venv_racketvision\Scripts\python.exe scripts/run_racketvision_worker.py data/raw/20251212_132025_1.mp4 data/processed/arrays/20251212_132025_1__racketvision__cache.jsonl
.venv_racketvision\Scripts\python.exe scripts/run_racketvision_worker.py data/raw/20251212_140101_1.mp4 data/processed/arrays/20251212_140101_1__racketvision__cache.jsonl
.venv\Scripts\python.exe scripts/process_ball_tracking_real.py
.venv\Scripts\python.exe -m pytest -q
```

`config/racketvision_manifest.json` fija el commit upstream, versiones y hashes
de los tres checkpoints. El worker escribe JSONL normalizado y un manifiesto con
hash de video, caché y pesos; el consumidor rechaza una caché alterada. Código,
pesos, entorno y cachés permanecen ignorados por Git.

El MP4 puede omitirse con `--no-overlay`. Activarlo no modifica la detección: el
dibujo opera sobre una copia del frame. Dos ejecuciones sin overlay del clip
fácil produjeron el mismo SHA-256 para su archivo `*__ball_tracking__frames.csv`.

## Límites pendientes

La cámara registra cerca de 30 fps y la bola recorre decenas de píxeles durante
una exposición; centro y velocidad tienen incertidumbre temporal/espacial. La
mesa se marcó de forma aproximada. Ambas raquetas se siguen, pero la pala negra
requiere a menudo el proxy de muñeca y la del jugador ocluido pierde confianza.
Las anotaciones existentes no son exhaustivas y no permiten todavía medir PCK,
precisión global de eventos ni decidir fine-tuning. Hace falta ampliar anotación
independiente, calibrar cámara/mesa y adquirir más fps antes de informar m/s,
drag ajustado o spin efectivo.
