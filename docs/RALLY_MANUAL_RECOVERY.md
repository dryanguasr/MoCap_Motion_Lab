# Intercambios y recuperación manual

Este flujo trabaja sobre vistas del video original delimitadas por fotogramas.
Propone intervalos, permite revisarlos y exporta MP4 sin cambiar el encuadre ni
interpolar fotogramas. Las propuestas requieren revisión humana antes de aceptarlas.
Las mediciones siguen siendo aparentes en px/s.

## Inicio en Windows

1. Abrir `review_professionals.cmd`: prepara el manifiesto si falta, abre la
   revisión guiada y al completarla exporta los intercambios aceptados con mesa validada.
   «Guardar y salir» deja la exportación para cuando termine la revisión.
2. Abrir `track_professionals.cmd`: por ahora procesa **solo el primer intercambio**
   (`rally_001_part_01`) y reanuda sus correcciones guardadas. Abre la inspección
   cuando se pierde la bola. Las correcciones se guardan al confirmarlas.
   El progreso mostrará 1/1 y el tiempo pendiente de este clip. Al terminar se
   detiene; los demás intercambios quedan fuera hasta revisar el diagnóstico.

Las etapas pueden ejecutarse por separado desde la raíz del proyecto:

Los comandos generales siguientes no tienen la restricción del lanzador piloto.
Para mantener la prueba limitada, añada `--clip-id rally_001_part_01` al procesador.

```powershell
.venv\Scripts\python.exe scripts/prepare_rally_clips.py data/raw/Ma-Long-and-Fan-Zhendong-Training-T2-Diamond-2019-Malaysia.mp4 --manifest data/annotations/rallies/professionals.json
.venv\Scripts\python.exe scripts/prepare_rally_clips.py --manifest data/annotations/rallies/professionals.json --review
.venv\Scripts\python.exe scripts/prepare_rally_clips.py --manifest data/annotations/rallies/professionals.json --export
.venv\Scripts\python.exe scripts/process_ball_tracking_real.py --manifest data/annotations/rallies/professionals.json --interactive
```

`--clip-id rally_003_part_01` limita el procesamiento a una parte; es repetible.
`--interventions-dir` selecciona otra sesión de correcciones. Sin `--interactive`,
se reproducen las correcciones existentes y se guarda `needs_inspection` ante una
decisión pendiente. `--no-overlay` omite el MP4 diagnóstico. Sin `--manifest`, el
procesador original conserva su funcionamiento.

## Revisión de intervalos y mesa

El modo habitual presenta una pregunta por pantalla, con botones. La barra permite
buscar un momento; «Reproducir lento» y «Anterior/Siguiente frame» permiten inspeccionarlo.

1. **Cambio de cámara:** muestra contexto antes y después del corte propuesto.
   Si es real, pide señalar el primer fotograma del nuevo encuadre; si no, une las tomas.
2. **Intercambio pendiente:** pregunta si hay un solo intercambio completo. «Está completo»
   lo acepta. «Corregir límites» pide primero el primer fotograma a conservar y luego
   el último, ambos incluidos. «Hay varios» pide dónde separar. «Sigue en el próximo»
   une con la siguiente propuesta. «No hay juego» elimina esa propuesta.
3. **Mesa y malla:** aparece automáticamente para las tomas utilizadas. Muestra las
   líneas existentes sobre inicio, centro y final. Basta responder «Sí, coinciden».
   Si elige «Corregir puntos», haga clic en el punto dibujado que está mal y luego
   en su posición correcta. Pulse S para volver a comprobar las tres imágenes.
   Sin propuesta previa, la pantalla pide cada esquina por su nombre, en español.

Las respuestas se guardan y las comprobaciones resueltas no se repiten al reabrir.
La comprobación de mesa se comparte entre los clips de la misma toma. Los borradores
de puntos se guardan al salir y después de confirmar cada imagen de referencia.
Si sale a mitad de ajustar límites, se conserva el intervalo anterior completo.
Las propuestas actuales son heurísticas y no tienen una confianza calibrada:
todos los intercambios aún no aceptados necesitan esa primera comprobación.

### Editor avanzado opcional

Solo si necesita edición libre, ejecute `--advanced-review` en lugar de `--review`.
Los siguientes atajos corresponden exclusivamente a ese editor:

| Tecla | Acción |
|---|---|
| Espacio; V | Reproducir/pausar; cambiar entre 0,25x, 0,5x y 1x |
| A / D; J / L | Un fotograma atrás/adelante; un segundo atrás/adelante |
| N / P | Seleccionar intercambio siguiente/anterior |
| I / O | Fijar inicio/final en el fotograma mostrado |
| C / X | Añadir intervalo mínimo de tres fotogramas / eliminar seleccionado |
| T / M | Dividir en el fotograma actual / unir con el siguiente |
| Enter | Aceptar o desmarcar el intercambio seleccionado |
| B / H | Añadir cambio de toma / unir toma actual con la anterior |
| G | Revisar mesa y malla de la toma actual |
| S / Q | Guardar / guardar y salir |

Los cambios de límites se guardan inmediatamente y desmarcan su aceptación.
No se permiten solapamientos ni partes de menos de tres fotogramas. Cambiar una
toma invalida su geometría. Un intercambio que cruza tomas conserva un mismo
`rally_id` y produce varias partes.

En el anotador, `1`, `2` y `3` muestran inicio, centro y final; `0` muestra el
fondo mediano. Visitar los tres fotogramas después del último cambio de puntos
antes de guardar con `S`. `U` deshace y `R` reinicia los ocho puntos. Clic indica
visible y Shift+clic estimado por oclusión. Si la mesa cambia entre las vistas,
salir y dividir la toma con `B`. La anotación del video largo es solo una propuesta.

Los cambios de toma se proponen cada 0,5 s usando apariencia y registro visual
del fondo/mesa, excluyendo logos y parte de las regiones de jugadores. Los límites
son aproximados y se ajustan fotograma a fotograma. No se estabiliza la imagen.

La propuesta de intercambios combina movimiento de bola y raqueta/muñeca; exige
actividad sostenida para evitar que detecciones aisladas o gestos de preparación
unan todo el video. Valores iniciales: `--activity-window-s 0.8`,
`--activity-fraction 0.3`, `--inactivity-s 1.5`, `--padding-s 0.5`. Se registran
en el manifiesto y no hay duración máxima obligatoria. Son criterios heurísticos,
no una clasificación automática fiable de todos los puntos.

## Inspección de la bola

La ventana se abre al llegar a `LOST`, respetando la ventana de predicción, y
muestra el primer fotograma sin observación. Si nunca se inicializó, aparece tras
0,5 s. Se agrupan las pérdidas por episodio. La semilla no cuenta como detección.

La interfaz se maneja con botones; no hace falta memorizar teclas:

La barra inferior muestra el intercambio actual sobre el total, el porcentaje
recorrido del clip y del conjunto, los segundos de vídeo pendientes y cuántos
intercambios vienen después. Usa los timestamps reales y el inicio del episodio
pendiente: avanzar el cursor para buscar la bola no aumenta el progreso.
El porcentaje mide recorrido procesado, no exactitud ni porcentaje de bola bien
identificada. Los segundos pendientes son duración de vídeo, no tiempo estimado
de trabajo humano. La línea verde se interrumpe en pérdidas y reinicios manuales.

1. Haga clic sobre la bola en la imagen o en la lupa lateral. El clic pausa
   la reproducción para conservar exactamente el fotograma señalado.
2. Compruebe la cruz rosa y pulse **Confirmar y continuar**. Puede cambiar el
   punto haciendo otro clic o pulsando **Corregir punto**.
3. Si no aparece la bola, pulse **No encuentro la bola**. Puede estimar su
   posición oculta, omitir un tramo o terminar el intercambio. Para omitir o
   terminar, la pantalla pide buscar el último fotograma y confirmar la acción.

Los botones de navegación permiten avanzar/retroceder un fotograma o 0,2 s y
reproducir a media velocidad. **Visible / Estimada (cambiar tipo)** cambia cómo
se interpreta el clic. La lupa sigue el cursor y no tapa el vídeo.
**Guardar y salir** conserva el progreso confirmado y el fotograma para reanudar.
Un punto pendiente solo se guarda al pulsar **Confirmar y continuar**; navegar
a otro fotograma lo descarta para evitar aplicarlo a una imagen diferente.

Los atajos A/D, J/L, Espacio, V/E, U, Enter y Q siguen disponibles. S/F abren
la selección del final del tramo a omitir/intercambio y requieren confirmación
con el botón correspondiente.

La recuperación reinicia vuelo e hipótesis de interacción. Un clic no determina
velocidad: se requieren dos detecciones visuales posteriores coherentes entre sí
y con la posición marcada. Las semillas ocluidas admiten mayor incertidumbre.
Durante la recuperación manual se amplía el límite de longitud de blur del
detector clásico, sin modificar sus parámetros automáticos habituales.

Si no se confirma en 0,25 s, reaparece la inspección para corregir, buscar otra
imagen, omitir el hueco o terminar. Después de una semilla no se adopta la
trayectoria clásica previa. Suavizado y eventos respetan los reinicios.
La identidad recuperada debe comprobarse visualmente: dos detecciones coherentes
todavía pueden corresponder a un distractor.

## Persistencia, sincronización y salidas

- `seima.rallies.v1`: fuente/hash, PTS de todos los fotogramas, referencias y hashes
  de cachés, tomas, intercambios y mapas explícitos de clips. Los intervalos son
  `[start_frame, end_frame)`; O incluye el fotograma mostrado.
- `seima.ball-interventions.v1`: identidad del clip y geometría/cachés, acciones,
  historial reemplazado, coordenadas, tipo, fotogramas locales/originales y tiempos.
- Cada corrección se guarda mediante reemplazo atómico. Editar hacia atrás invalida
  decisiones posteriores dependientes y conserva su historial. Reanudar recalcula
  el clip con las correcciones; no repite inferencia neuronal. La evidencia clásica
  se prepara una vez por clip y ejecución.
- Las salidas usan `ball_tracking_assisted`: métricas CSV, eventos JSON, resúmenes
  y overlays. No sobrescriben los resultados automáticos `ball_tracking` anteriores.
  Separan semillas visibles/estimadas, recuperaciones, observaciones y predicciones.
- El motor lee el original, no el MP4 recodificado. Las cachés se seleccionan por
  índice original y se ajustan a sus PTS. No se renombran como si provinieran de
  los clips nuevos. La cobertura faltante al final se representa sin evidencia.
- Los saltos de revisión e inspección se realizan con FFmpeg y PTS absolutos.
  No se usa `CAP_PROP_POS_FRAMES` para buscar imágenes en el video VFR: OpenCV
  puede devolver una imagen equivocada aunque reporte el índice solicitado.
  La decodificación desde el inicio sigue siendo secuencial.
- La caché histórica de profesionales usaba `frame/fps` pese a los PTS variables.
  Se reconoce ese reloj comparándolo con los fps del contenedor, se verifica el
  hash del video y se conservan índices contiguos; los tiempos anidados se sustituyen
  por PTS originales. Los relojes desconocidos se rechazan.
- Las poses antiguas no incluyen hash original. Se comprueban con resumen de
  origen, dimensiones y número de frames; se identifican como procedencia histórica
  y se fijan hashes desde la preparación. No es prueba retroactiva de su generación.
- Los MP4 de análisis son **sin audio**. Los clips usan H.264 sin pérdida; los
  overlays usan compresión de visualización. Se verifican todos los PTS, resolución
  y decodificación del último fotograma. Nunca se sobrescribe el video fuente.

FFmpeg/FFprobe se resuelven por opciones `--ffmpeg`/`--ffprobe`, variables
`SEIMA_FFMPEG`/`SEIMA_FFPROBE`, PATH, `.cache/video-tools/` o WinGet. No se instala
software automáticamente ni se usa fallback de fps en este flujo.
Si cambian geometría, límites o cachés, usar otra sesión de intervenciones;
una sesión incompatible se rechaza explícitamente.

## Validación

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Las pruebas cubren recuperación, clic incorrecto, oclusión, conflictos con ancla
clásica, pérdidas, replay, persistencia, coordenadas de zoom, edición de intervalos,
geometría/procedencia y exportación VFR. La aceptación de todos los intercambios y
la identidad tras cada recuperación requieren revisión humana; los resúmenes
conservan ese estado pendiente.
