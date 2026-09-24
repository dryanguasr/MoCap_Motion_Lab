# SEIMA-MoCap

Base experimental para **captura y análisis markerless de movimiento humano a partir de video**, pensada como plataforma común para proyectos de semillero en Ingeniería Mecatrónica.

La primera versión usa **MediaPipe Pose Landmarker** para detectar 33 puntos corporales en video de webcam, estimar posiciones 3D, aplicar un filtrado temporal básico y calcular velocidades lineales. Además, reconstruye un conjunto inicial de ángulos geométricos articulares y sus velocidades angulares.

> **Estado:** prototipo base / etapa exploratoria. Los resultados son útiles para experimentación, desarrollo de algoritmos y comparación relativa de movimientos, pero todavía no deben interpretarse como mediciones biomecánicas clínicas ni como un sistema de MoCap metrológicamente validado.

## Objetivo general

Desarrollar una plataforma modular de bajo costo que permita extraer, procesar y analizar información cinemática del movimiento humano a partir de video convencional, facilitando que distintos estudiantes desarrollen aplicaciones específicas sobre una base técnica común.

## Alcance inicial

- Captura de video desde la cámara del computador.
- Detección markerless de una persona mediante MediaPipe Pose Landmarker.
- Visualización del esqueleto corporal sobre el video.
- Recuperación de los 33 landmarks corporales.
- Uso de coordenadas 3D estimadas por MediaPipe en metros.
- Suposición inicial de **dominancia derecha** cuando no exista información adicional.
- Seguimiento específico de la muñeca derecha:
  - posición 3D;
  - velocidad 3D;
  - magnitud de la velocidad.
- Filtrado causal básico mediante media móvil exponencial (EMA).
- Estimación inicial de ángulos geométricos en hombros, codos, caderas y rodillas.
- Estimación de velocidades angulares por diferencia finita sobre señales filtradas.
- Registro opcional de datos en CSV para análisis posterior.

## Casos de uso

La plataforma **no está cerrada a una única aplicación**. La línea personal del director incluye análisis de movimiento en tenis de mesa, pero la base se plantea para que los estudiantes puedan abordar, entre otros:

- análisis deportivo;
- ergonomía;
- agricultura y actividades productivas;
- rehabilitación y movimiento funcional;
- interacción humano-robot;
- evaluación de posturas y gestos técnicos.

## Arquitectura

```text
Webcam / video
      │
      ▼
MediaPipe Pose Landmarker
      │
      ├── landmarks normalizados 2D/3D de imagen
      └── landmarks 3D de mundo
                │
                ▼
       Filtrado temporal básico
                │
                ├── posiciones y velocidades lineales
                └── ángulos y velocidades articulares aproximadas
                │
                ▼
       Visualización + registro CSV
```

Consulta [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) para más detalle.

## Requisitos recomendados

- Windows o Linux/WSL con acceso a webcam.
- Python **3.10, 3.11 o 3.12**.
- Webcam integrada o USB.

MediaPipe dispone actualmente de Pose Landmarker como parte de MediaPipe Tasks. El repositorio fija la dependencia a la serie 1.x para evitar cambios mayores inesperados.

## Instalación rápida

### 1. Crear un entorno virtual

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / WSL:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar el proyecto

```bash
python -m pip install --upgrade pip
pip install -e .
```

### 3. Descargar el modelo

```bash
python scripts/download_model.py
```

Por defecto se descarga `pose_landmarker_full.task` en `models/`.

### 4. Ejecutar la webcam

```bash
seima-mocap
```

También puede ejecutarse con:

```bash
python -m seima_mocap.cli
```

Presiona `q` o `ESC` para cerrar.

## Registrar una sesión

```bash
seima-mocap --record
```

Se crean dos archivos en `data/processed/metrics/`, identificados con el nombre
de sesión y el pipeline `capture`:

- `session_<fecha>__capture__landmarks.csv`: posiciones y velocidades 3D;
- `session_<fecha>__capture__joints.csv`: ángulos y velocidades angulares.

Los videos crudos y archivos de datos no se versionan por defecto.

## Pilotos con videos de tenis de mesa

Los flujos experimentales añadidos permiten procesar una pose principal, dos
jugadores y lotes centrados en el jugador situado a la izquierda:

```powershell
python scripts/process_training_video.py --help
python scripts/process_two_player_video.py --help
python scripts/analyze_left_player_batch.py --help
```

El piloto de cinética planar reutiliza las poses ya calculadas de seis clips
cortos y ejecuta escenarios de sensibilidad para fuerzas proyectadas y momentos
netos de tobillo, rodilla y cadera:

```powershell
python scripts/estimate_planar_kinetics.py
```

Consulta [`docs/PROTOCOLO_ADQUISICION_CINETICA.md`](docs/PROTOCOLO_ADQUISICION_CINETICA.md)
para el protocolo propuesto con cámaras calibradas, plataformas de fuerza y
antropometría. Los resultados se organizan por tipo bajo `data/processed/`; el
informe está en `data/processed/reports/batch__planar_kinetics__report.md`.
Son una instantánea exploratoria: **no son mediciones de fuerza ni torques
anatómicos validados**. Los videos y poses de entrada permanecen excluidos.

El seguimiento de bola dispone también de una canalización multimodal offline:
combina el detector clásico, BallTrack, pose corporal de ambos jugadores y
cinco puntos por raqueta. RacketVision se ejecuta en un entorno Python 3.10
aislado y sus cachés se validan mediante manifiesto y SHA-256:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_racketvision.ps1
.venv_racketvision\Scripts\python.exe scripts/run_racketvision_worker.py data/raw/CLIP.mp4 data/processed/arrays/CLIP__racketvision__cache.jsonl
.venv\Scripts\python.exe scripts/process_ball_tracking_real.py CLIP
```

Si no existe la caché, continúa con el detector clásico y una geometría de baja
confianza basada en muñeca–antebrazo. Consulta
[`docs/BALL_TRACKING_REAL.md`](docs/BALL_TRACKING_REAL.md).

Para dividir el video de profesionales por intercambios y recuperar la bola con
semillas manuales, abre `review_professionals.cmd` y después
`track_professionals.cmd`. La revisión permite ajustar cortes, validar la mesa por
toma y aceptar intervalos. El seguimiento se pausa al perder la bola y guarda las
correcciones para reanudarlas. Consulta
[`docs/RALLY_MANUAL_RECOVERY.md`](docs/RALLY_MANUAL_RECOVERY.md) para los controles.

Todas las salidas usan el patrón `fuente__pipeline__artefacto.ext`. Consulta
[`data/processed/README.md`](data/processed/README.md) para la estructura de
videos, métricas, eventos, datasets, resúmenes, arrays y diagnósticos.

## Parámetros útiles

```bash
seima-mocap --help
```

Ejemplos:

```bash
# Usar otra cámara
seima-mocap --camera 1

# Cambiar intensidad del suavizado
seima-mocap --alpha 0.20

# Exigir mayor visibilidad del landmark
seima-mocap --min-visibility 0.70

# Registrar datos
seima-mocap --record
```

## Qué significa “posición/velocidad articular” en esta versión

Se manejan dos niveles de información:

1. **Landmarks:** posiciones cartesianas 3D de puntos anatómicos estimados por MediaPipe y sus velocidades lineales.
2. **Articulaciones:** ángulos geométricos definidos por tres landmarks y sus velocidades angulares.

Estos ángulos son descriptores cinemáticos útiles, pero **no equivalen todavía a una reconstrucción biomecánica completa de los grados de libertad articulares**. Por ejemplo, el hombro real tiene varios grados de libertad y un único ángulo entre brazo y tronco no los representa completamente.

## Muñeca dominante

En ausencia de información adicional se asume una persona diestra, por lo que el punto de interés principal es `RIGHT_WRIST` (landmark 16). Esta decisión está aislada en la configuración para que posteriormente pueda:

- preguntarse al usuario;
- inferirse la dominancia;
- analizarse ambas extremidades de forma simultánea.

## Limitaciones metodológicas actuales

- Una sola cámara genera oclusiones y ambigüedad de profundidad.
- Las coordenadas 3D de MediaPipe son estimadas por un modelo, no trianguladas con cámaras calibradas.
- La velocidad es sensible al ruido y a variaciones en la tasa efectiva de cuadros.
- Los ángulos calculados son geométricos y no representan todavía sistemas de coordenadas anatómicos completos.
- No hay por ahora calibración intrínseca/extrínseca de cámara.
- No existe aún validación contra un sistema de referencia.

Estas limitaciones son parte explícita de la agenda de investigación y no deben ocultarse al reportar resultados.

## Próximos pasos sugeridos

1. Validar repetibilidad estática y dinámica.
2. Comparar filtros: EMA, Butterworth, Savitzky-Golay y One Euro.
3. Procesar archivos de video además de webcam.
4. Añadir calibración de cámara y referencias espaciales conocidas.
5. Evaluar MediaPipe Holistic o Hand Landmarker para orientación detallada de mano y dedos.
6. Definir modelos cinemáticos específicos por aplicación.
7. Incorporar métricas de calidad/confianza y manejo de oclusiones.
8. Validar trayectorias frente a referencias conocidas o un sistema MoCap externo.
9. Construir módulos específicos, por ejemplo análisis de golpes de tenis de mesa.

## Estructura del repositorio

```text
SEIMA-MoCap/
├── README.md
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── ARCHITECTURE.md
│   └── METHODOLOGY_NOTES.md
├── models/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   └── download_model.py
├── src/
│   └── seima_mocap/
│       ├── __init__.py
│       ├── cli.py
│       ├── filters.py
│       ├── kinematics.py
│       ├── landmarks.py
│       └── recorder.py
└── tests/
    └── test_kinematics.py
```

## Criterio de desarrollo

El repositorio busca separar claramente:

- **detección** de landmarks;
- **filtrado** de señales;
- **cinemática**;
- **visualización**;
- **registro de datos**;
- **aplicaciones específicas**.

Esto permitirá reemplazar MediaPipe por otra librería en el futuro sin reescribir todo el proyecto.
