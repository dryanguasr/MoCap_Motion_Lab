# Piloto computacional de fuerzas y momentos — seis clips cortos

**Resultados condicionales de un modelo 2D, NO mediciones de fuerza ni torques anatómicos validados.**

Analizados 1580 frames del jugador principal situado a la izquierda. El video largo `20251212_132025.mp4` queda excluido y no se ha borrado ni alterado. Izquierdo/derecho en las variables identifica las extremidades del jugador principal, no al oponente.

El escenario base admite fuerza total en 785/1580 frames (49.7 %) y fuerzas por pie/momentos en 468/1580 (29.6 %). Cobertura significa pasar filtros internos, NO exactitud o confianza estadística. Las regiones omitidas no son aleatorias: no se pueden generalizar estos resúmenes a todos los golpes.

## Resultados base

Fz es la componente vertical **supuesta en el plano de imagen**, no una plataforma de fuerza. Los percentiles usan exclusivamente frames admitidos; no son picos reales del gesto.

| Clip | Fuerza admitida | Momentos admitidos | Mediana Fz (N) | P95 Fz (N) |
|---|---:|---:|---:|---:|
| 20251212_132025_1.mp4 | 58.3 % | 39.7 % | 870 | 1129 |
| 20251212_133639_1.mp4 | 48.7 % | 25.7 % | 866 | 1150 |
| 20251212_134838_1.mp4 | 31.3 % | 18.7 % | 871 | 1058 |
| 20251212_135118_1.mp4 | 58.2 % | 34.8 % | 883 | 1193 |
| 20251212_135431_1.mp4 | 45.8 % | 23.1 % | 894 | 1172 |
| 20251212_140101_1.mp4 | 50.2 % | 31.1 % | 888 | 1243 |

Magnitud P95 del momento neto proyectado (N·m), no torque muscular ni indicador de lesión:

| Clip | Tobillo I / D | Rodilla I / D | Cadera I / D |
|---|---:|---:|---:|
| 20251212_132025_1.mp4 | 50.4 / 40.5 | 65.6 / 99.8 | 157.5 / 143.5 |
| 20251212_133639_1.mp4 | 53.8 / 34.4 | 49.8 / 84.3 | 122.0 / 136.9 |
| 20251212_134838_1.mp4 | 37.6 / 46.4 | 35.0 / 55.5 | 146.9 / 75.0 |
| 20251212_135118_1.mp4 | 42.4 / 32.9 | 62.2 / 124.3 | 120.6 / 168.4 |
| 20251212_135431_1.mp4 | 60.0 / 57.8 | 91.3 / 111.1 | 174.8 / 160.9 |
| 20251212_140101_1.mp4 | 33.5 / 52.8 | 74.9 / 82.9 | 60.3 / 208.9 |

## Hipótesis y ecuaciones

- Masa corporal 90 kg (peso 882.9 N), talla 1.84 m; raqueta supuesta de 0.18 kg como masa puntual en la muñeca derecha. El sistema completo pesa 884.67 N en reposo. No hay medición de raqueta ni datos de InBody incorporados.
- Se utilizan x/y de imagen, NO las coordenadas world de MediaPipe centradas en pelvis. Eje x a la derecha, z hacia arriba en imagen, gravedad asumida paralela a ese eje. Se ignoran profundidad, inclinación real de cámara, rotación fuera del plano y perspectiva. No es un modelo sagital 3D.
- Una escala FIJA por clip: 0.288 × 1.84 m dividido por la mediana del torso proyectado en frames fiables. Ese 28.8 % es un supuesto heredado, no una medida anatómica. No derivamos una escala variable frame a frame. Origen fijo; la altura del COM es proyectada respecto a un suelo aproximado y no altura física calibrada.
- 14 segmentos corporales: tronco 43.46 %, cabeza/cuello 6.94 %; POR LADO brazo 2.71 %, antebrazo 1.62 %, mano 0.61 %, muslo 14.16 %, pierna 4.33 %, pie 1.37 %. Priors de referencia masculina inspirados en de Leva, no ajuste individual ni confirmación de su adecuación al sujeto. La suma se normaliza a la masa corporal.
- COM supuesto a fracción proximal→distal: tronco 0.50, cabeza/cuello 0.70 hacia nariz desde hombros, brazo 0.45, antebrazo 0.43, mano 0.50, muslo 0.41, pierna 0.44, pie 0.50 talón→punta. La mano se extrapola 20 % de la longitud de antebrazo. Son simplificaciones explícitas, NO una tabla completa de de Leva.
- Inercias planares de barras homogéneas I=m×L_mediana²/12. Sin inercia rotacional de raqueta. No hay tensores inerciales 3D, masas segmentarias medidas ni estimación de músculo individual.
- Ajuste polinomial local cúbico centrado de aproximadamente 0.4 s (ventana impar según fps), usando timestamps reales de ffprobe. Primera/segunda derivadas analíticas del ajuste. No se interpolan ventanas rechazadas.

Balance global: F_suelo = Σ m_i (a_i − g); COM de cuerpo + raqueta. dH_COM/dt = Σ[I_i α_i + (c_i − COM) × m_i a_i]. Se suponen nulas otras fuerzas externas.

Reparto entre pies: se supone doble apoyo, COP de cada pie en el 50 % talón→punta, momento libre nulo y fuerzas izquierda/derecha paralelas. F_I=λF, F_D=(1−λ)F. Se resuelve λ con el balance de momento global; sin estas restricciones no hay una solución única. COP es un punto proyectado SUPUESTO, no medido ni limitado por una huella real. No se calcula una alternativa de apoyo simple para completar huecos.

Recursión distal→proximal en pie, pierna y muslo: F_prox=m(a−g)−F_dist; τ_prox=Iα−τ_dist−(r_dist−c)×F_dist−(r_prox−c)×F_prox. En el siguiente segmento se invierten fuerza y momento por acción/reacción. Signo positivo: r_x F_z − r_z F_x, antihorario en el gráfico x/z; momento que actúa sobre el segmento distal. No interpretar el signo como flexión/extensión o abducción anatómica.

## Filtros de admisión (heurísticos, no umbrales clínicos)

1. Coordenadas finitas y puntos usados dentro de un margen de imagen de 0.5 %. Pelvis del objetivo en x<0.58. Calidad=min(visibility,presence): tronco/rodillas/tobillos ≥0.4; talones/puntas ≥0.3; nariz/codos/muñecas mínimo ≥0.1 y promedio ≥0.5. Longitudes proyectadas de torso, muslos y piernas entre 0.6 y 1.4 veces su mediana.
2. Todos los frames de la ventana de derivación deben ser admisibles. Se descartan bordes y discontinuidades temporales >1.75 veces dt mediano. Se excluye ±0.10 s alrededor de los picos semánticos de muñeca previos, y toda ventana que los toque. Estos son candidatos de golpe, NO detecciones verificadas de contacto de pelota; pueden quedar impactos no detectados. Tampoco se detectan automáticamente apoyos de mano en mesa.
3. Fuerza total: 0.2≤Fz/peso_corporal≤2.5 y |Fx|≤μFz, con μ=0.7 supuesto. No se recortan valores a esos límites. La fricción real y la vertical real no están medidas.
4. Para reparto/torques: fracción de carga 0≤λ≤1 sin clipping, brazo proyectado efectivo entre apoyos ≥0.06 m. Ambos pies deben ser compatibles con cuasi-apoyo: |velocidad vertical media talón/punta|≤0.45 m/s, velocidad proyectada≤1 m/s y altura≤0.15 m por encima del P20 del pie correspondiente. Es consistencia con un supuesto, no un detector validado de contacto.
5. Residuo de momento del reparto <10⁻⁶ N·m como comprobación algebraica. Ese cierre se obtiene por construcción y no demuestra fidelidad biomecánica.

## Sensibilidad e interpretabilidad

Quince escenarios: base y variaciones UN FACTOR A LA VEZ de masa 85/95 kg, escala ±10 %, ventana 0.3/0.5 s, COP 20/80 % talón→punta, μ=0.5/0.9, masas de extremidades ±10 % renormalizando a masa corporal, e inercias ±20 %. Son escenarios elegidos para explorar sensibilidad, no distribuciones calibradas de error.

Las bandas muestran mínimo/máximo de escenarios admitidos en cada instante con base admitida. Se registra cuántos escenarios sobreviven: cuando desaparece un escenario NO sabemos qué habría ocurrido allí. No son intervalos de confianza, no combinan simultáneamente perturbaciones y no cubren todos los errores de profundidad, COM, contacto, pose o cámara. Una banda estrecha de fuerza no implica una medida precisa.

La cobertura baja y la sensibilidad de los momentos al COP impiden conclusiones sobre asimetrías, sobrecarga o riesgo de lesión. Medianas cercanas al peso corporal son compatibles con gravedad dominante, no validación. El muestreo cercano a 30 fps y el suavizado excluyen resolver picos rápidos o impactos.

No se estiman torques de muñeca/codo/hombro en este piloto: la raqueta se incluye solo como masa puntual para el balance corporal. Faltan orientación/inercia de la raqueta y carga de impacto para un análisis defendible.

## Archivos y reproducción

- `batch_summary.csv/json`: resumen; `assumptions.json`: escenarios; `manifest.json`: fuentes, SHA-256 y versiones.
- Por clip: `frame_kinetics.csv` (unidades en nombres, 0/1 en flags, vacío=sin estimación), `sensitivity_envelope.csv`, `scenario_series.npz`, `scenario_summaries.json`, `summary.json` y `kinetics_diagnostics.png`.
- Los campos `diagnostic` pueden contener hipótesis inadmisibles; no son resultados cinéticos aceptados. Los campos COM describen el modelo proyectado; `total_force_valid` y `torque_valid` gobiernan la admisión.
- Ejecutar desde la raíz del repositorio: `.venv/Scripts/python scripts/estimate_planar_kinetics.py`. Requiere numpy, matplotlib y ffprobe; reutiliza poses existentes. No necesita OpenSim ni nuevas dependencias instaladas.
- Tests analíticos en `tests/test_planar_dynamics.py`: balance estático/dinámico, signos, reparto inadmisible y derivación con timestamps irregulares/huecos. Son controles de implementación, NO validación experimental.

Protocolo para datos reales: `docs/PROTOCOLO_ADQUISICION_CINETICA.md`. Los datos y resultados locales siguen excluidos de Git por las reglas existentes; no se han subido ni creado commits.

## Referencias

[OpenSim: entradas y alcance de dinámica inversa](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090063). [De Leva: parámetros antropométricos de referencia](https://pubmed.ncbi.nlm.nih.gov/8872282/). La implementación es un modelo planar propio simplificado, no una ejecución de esos paquetes ni una réplica completa del artículo.
