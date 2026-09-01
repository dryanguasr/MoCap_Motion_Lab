# Arquitectura inicial

## Principio de diseño

La plataforma debe permitir que la tecnología de estimación de pose sea reemplazable. MediaPipe es el punto de partida, no una dependencia conceptual permanente del proyecto.

## Flujo de datos

1. **Adquisición**
   - webcam en tiempo real;
   - posteriormente: archivos de video y cámaras externas.
2. **Estimación de pose**
   - MediaPipe Pose Landmarker;
   - 33 landmarks;
   - coordenadas normalizadas para visualización;
   - coordenadas de mundo para procesamiento cinemático.
3. **Control de calidad**
   - umbral de visibilidad;
   - posteriormente: presencia, continuidad, detección de outliers y oclusiones.
4. **Filtrado**
   - EMA causal por landmark;
   - EMA adicional sobre velocidades.
5. **Cinemática cartesiana**
   - posición 3D;
   - velocidad por diferencia finita usando timestamps reales.
6. **Cinemática articular inicial**
   - ángulos entre segmentos definidos por tripletas de landmarks;
   - velocidad angular por diferencia finita.
7. **Presentación y persistencia**
   - overlay sobre webcam;
   - CSV de landmarks;
   - CSV de articulaciones.

## Separación de responsabilidades

### `landmarks.py`

Define nomenclatura, índices, conexiones visuales y tripletas articulares. No procesa video.

### `filters.py`

Contiene filtrado temporal y derivación. Debe poder reutilizarse independientemente del detector de pose.

### `kinematics.py`

Convierte posiciones de landmarks en descriptores articulares.

### `recorder.py`

Gestiona persistencia de datos experimentales sin mezclarla con la detección.

### `cli.py`

Orquesta webcam, MediaPipe, visualización y los módulos anteriores. A futuro conviene extraer MediaPipe a una interfaz `PoseBackend` para soportar otras librerías.

## Extensiones previstas

```text
                         ┌─ MediaPipe Pose
Video → PoseBackend ─────┼─ MediaPipe Holistic
                         ├─ MMPose
                         └─ otro modelo
                               │
                               ▼
                    LandmarkFrame estándar
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
              filtros      cinemática     calidad
                 │             │             │
                 └─────────────┴─────────────┘
                               │
                               ▼
                         aplicaciones
```

## Decisiones de la versión 0.1

- Se usa `VIDEO` mode de Pose Landmarker incluso con webcam para mantener un flujo síncrono sencillo y reproducible.
- Se procesa una sola persona.
- La muñeca dominante inicial es la derecha.
- Se privilegia claridad pedagógica sobre optimización prematura.
- No se introducen todavía filtros complejos ni reconstrucción biomecánica multisegmento.
