# Ajuste antropométrico provisional: jugador principal de 1,84 m

## Estado verificado

Los siete videos, 5.344 frames, métricas por frame y 73 eventos heurísticos quedaron procesados. Se verificó que el número de frames de cada MP4 coincide con su CSV y JSON. Cada sesión conserva `pose_landmarks.npz` con landmarks de imagen y mundo para futuros ajustes. No se modificaron los videos originales.

## Cómo se incorporó la talla

La talla conocida es **1,84 m**. La proporción hombro–cadera se **supuso** igual a 0,288 de la talla: longitud de torso de referencia = **0,52992 m**. Esa proporción no se midió en el jugador ni se validó para este conjunto de videos.

- Escala visual por frame: 0,52992 m / longitud hombro–cadera proyectada en píxeles, suavizada.
- Velocidad visual: velocidad de la muñeca derecha en píxeles/s multiplicada por esa escala.
- Altura de cadera: separación vertical proyectada entre el centro de caderas y el punto de pie visible más bajo, multiplicada por la escala.
- Variante 3D: velocidades de MediaPipe reescaladas por la relación entre el torso supuesto y el torso 3D mediano.

**Son estimaciones ajustadas por talla, no una calibración métrica validada.** La talla por sí sola no resuelve perspectiva, profundidad, orientación del torso ni la distancia al suelo. Una proporción corporal distinta cambiará proporcionalmente los valores. La velocidad 2D proyectada y la velocidad 3D relativa al origen corporal tampoco representan exactamente la misma magnitud.

## Resultados actualizados

| Video | Cadera media estimada | Cadera P05–P95 | Velocidad muñeca P95 | Máximo bruto¹ | Eventos candidatos |
|---|---:|---:|---:|---:|---:|
| `20251212_132025_1.mp4` | 0,889 m | 0,813–0,943 m | 4,94 m/s | 7,57 m/s | 5 |
| `20251212_132025.mp4` | 0,899 m | 0,814–0,964 m | 3,37 m/s | 16,76 m/s¹ | 46 |
| `20251212_133639_1.mp4` | 0,894 m | 0,803–0,986 m | 4,73 m/s | 7,83 m/s | 3 |
| `20251212_134838_1.mp4` | 0,888 m | 0,835–0,950 m | 3,32 m/s | 6,63 m/s | 5 |
| `20251212_135118_1.mp4` | 0,905 m | 0,796–0,984 m | 4,20 m/s | 7,58 m/s | 5 |
| `20251212_135431_1.mp4` | 0,905 m | 0,818–1,000 m | 4,89 m/s | 7,79 m/s | 3 |
| `20251212_140101_1.mp4` | 0,905 m | 0,782–1,041 m | 4,40 m/s | 6,41 m/s | 6 |

¹ El máximo del video largo ocurre en el frame 1069, aproximadamente **35,632 s**. La revisión visual alrededor de ese instante confirma salida del jugador por el borde izquierdo y pérdida de localización anatómica. Se conserva el máximo bruto para trazabilidad, pero **no es una velocidad deportiva válida**. Los demás máximos no se han validado manualmente como contactos reales. P95 es menos sensible a picos aislados, aunque no elimina por sí solo todos los errores.

## Interpretación prudente

- Las medias estimadas de altura de cadera se sitúan entre 0,888 y 0,905 m. Diferencias de uno o dos centímetros no justifican concluir cambios técnicos: la incertidumbre del método no está cuantificada.
- En los clips cortos, el P95 de velocidad proyectada de muñeca queda entre 3,32 y 4,94 m/s.
- La variante 3D escalada por talla da P95 entre 2,39 y 4,48 m/s en los clips cortos. El acuerdo entre métodos no equivale a validación; ambos proceden del mismo modelo y comparten errores.
- La muñeca es un proxy del agarre. No se está midiendo la velocidad del centro o del extremo de la pala: la rotación de la raqueta puede producir velocidades diferentes.
- La altura de cadera no equivale a altura del centro de masas corporal.

## Calidad y semántica

La pose se detectó en el 100 % de frames de cinco clips, el 98,2 % del video largo y el 97,1 % del último clip. Estas tasas describen salidas del detector, no exactitud anatómica comprobada en todos los frames. La visibilidad/presencia de muñeca supera el umbral 0,50 en el 85,3–100 % de los frames según el video.

Los 73 eventos semánticos son picos de velocidad dirigidos hacia la derecha de la imagen, usados como candidatos a golpe hacia el oponente. Sus ventanas de preparación, aceleración, impacto proxy, seguimiento y recuperación son reglas temporales, no fases validadas. El ajuste por talla no cambió estas etiquetas.

Los CSV conservan señales suavizadas/interpoladas en algunos intervalos sin detección; deben filtrarse por `detected`, visibilidad y presencia antes de utilizarlas. La altura métrica también descarta valores fuera del intervalo predefinido 0,25–1,35 m. Por tanto, sus estadísticas describen los valores aceptados, no todos los frames.

## Archivos

- `batch_motion_summary.csv`: comparación actualizada.
- `semantic_events_all_videos.csv`: candidatos a golpe con valores ajustados.
- `batch_analysis.json`: resultados y parámetros.
- Los archivos se agrupan por tipo en `metrics/`, `summaries/`, `events/`, `arrays/` y `videos/`; cada nombre incluye la fuente, `left_player` y el tipo de artefacto.
- `peak_audit.jpg`: revisión visual del máximo espurio del video largo.

## Para mejorar la escala

Una medición real de longitud hombro–cadera y una toma erguida con cuerpo completo y pies visibles permitirían reemplazar la proporción supuesta. Para velocidades y alturas métricas más fiables se necesita además calibración de cámara, referencias en el plano relevante y validación de los landmarks, especialmente durante salidas de encuadre y oclusiones.
