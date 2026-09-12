# Región guiada por gestos: primer checkpoint

Estado: **pendiente de validación visual**, sin autorización automática para
continuar experimentos. Solo se ha probado una configuración H1 en tres ventanas
del primer intercambio. No se ha ejecutado el piloto completo con H1 ni otros
intercambios. El resultado no demuestra una mejora de identidad de la bola.

## Hipótesis y cambio implementado

La inspección del usuario indica pérdida en todos los impactos y seguimiento
principalmente desde Fan Zhendong hacia Ma Long después de la malla. Es evidencia
cualitativa: aún faltan contactos etiquetados para cuantificar esa asimetría.

H1 propone usar preparación y ejecución para localizar el frente del jugador y
las posibles salidas de la bola. Esta primera variante solo pondera suavemente
la confianza de candidatos existentes según muñeca, raqueta observada y movimiento
causal. Las fases son hipótesis, no contactos confirmados. Una pose incierta amplía
la región; sin pose no se penalizan candidatos. No elimina candidatos ni inventa
observaciones o velocidades, ni cambia física, asociación de contacto o semillas.

H2 propone el extremo delantero del borrón con dirección respaldada por evidencia.
Sigue sin implementarse: requiere confirmar identidad y acordar qué instante de
la exposición representa la posición anotada. La resta local específica también
queda pendiente; el sistema anterior ya usa diferencias de imágenes y fondo.

El módulo `src/seima_mocap/stroke_guidance.py` se conecta mediante un callback
opcional después de fusionar candidatos. El lanzador habitual no lo activa.
`config/stroke_guidance_stage1.json` fija ventanas y configuración;
`scripts/stroke_guidance_checkpoint.py` prepara, ejecuta y compara ambas variantes.

## Protocolo reproducible

Baseline congelado: `16bc34c1eec078400f868e0db6a29d7b4d42c223`, importado en un
proceso separado desde un archivo de Git. Ambas variantes reciben los mismos
paquetes de evidencia, verificados mediante SHA-256, y una única semilla original
por ventana. Las etiquetas posteriores se cargan solo al evaluar resultados.

| Caso | Uso | Procesamiento [inicio, fin) | Visualización | Semilla |
|---|---|---|---|---|
| A | Desarrollo | [17, 100) | [17, 100) | 17 |
| B | Desarrollo | [265, 310) | [265, 310) | 265 |
| C | Reservado | [298, 390) | [325, 390) | 298 |

El bloque reservado de C es [315, 390): su semilla es anterior. Sus etiquetas
no ajustaron esta configuración, pero ya se habían discutido históricamente;
no es un conjunto nuevo, ciego ni representativo. Tras inspeccionar estos
resultados, C dejaría de ser independiente si se usa para ajustar parámetros.
Permanecen reservados [100, 255) y [390, 481).

Se conserva la procedencia de cachés originales de pose y RacketVision y se
retiman por índice original a PTS reales. No se repite inferencia neuronal.
La preparación reutiliza la política previa de 17 muestras de fondo de la misma
toma validada, incluso fuera de las ventanas; no procesa seguimiento de otros
intercambios. No se modificaron vídeo, manifiesto, geometría o intervenciones.

La ejecución continúa diagnósticamente tras una petición de ayuda, conservando
pérdidas y peticiones, sin aportar más clics. Por ello menos peticiones no equivale
a menos intervenciones realmente necesarias.

## Resultado de esta única configuración

| Medida en etiquetas posteriores a la semilla | A | B | C |
|---|---:|---:|---:|
| Etiquetas evaluadas | 5 | 4 | 6 |
| Etiquetas dentro de la región propuesta | 4 | 3 | 4 |
| Candidato bruto a menos de 35 px | 5 | 4 | 6 |
| Observación seleccionada a menos de 35 px, baseline | 0 | 0 | 0 |
| Observación seleccionada a menos de 35 px, H1 | 0 | 0 | 0 |
| Fotogramas OBSERVED, baseline | 60 | 29 | 35 |
| Fotogramas OBSERVED, H1 | 65 | 36 | 35 |

Son etiquetas escasas de casos difíciles, no exactitud global ni continuidad
validada. Un candidato próximo tampoco prueba por sí solo que sea la bola.
H1 aumenta OBSERVED en A/B sin mejorar la cercanía a estas etiquetas; en C los
conteos de estados y resultados en etiquetas no cambian. La ponderación espacial
suave es insuficiente en esta prueba. La presencia de candidatos cercanos en los
15 puntos sugiere revisar asociación e identidad, sin descartar fallos de región,
fase, contraste o convención del borrón.

## Artefactos y comprobaciones

Los artefactos grandes permanecen locales en
`data/processed/h1_stage1/checkpoint_01/`: `protocol.json`, `checkpoint.json`,
`bundle_hashes.json`, paquetes de entrada/salida, `metrics.json`, trazas por
variante, `comparison_frame_map.json`, `comparacion_normal.mp4` y
`comparacion_lenta.mp4`. Requieren el vídeo y las anotaciones locales para repetir
la preparación; no se distribuyen esos datos mediante Git.

Los vídeos contienen 193 fotogramas comparados, resolución 1920 × 706. El normal
respeta diferencias de PTS dentro de cada ventana y omite los huecos entre ellas;
el lento multiplica los tiempos por cuatro. Ambos muestran índice y PTS del
original. No hay suavizado ni interpolación. Izquierda: baseline; derecha: H1.
Verde: observación; naranja: predicción; magenta: semilla; cian: región propuesta.
Los ocho candidatos blancos son evidencia bruta, no toda la lista fusionada;
la traza JSON registra los candidatos posteriores a la ponderación H1.

Se corrigió la base temporal de salida rawvideo de FFmpeg a microsegundos para
evitar advertencias DTS al buscar en VFR que podían bloquear su tubería de error.
Se reanudó solo la preparación pendiente de C; A/B se reutilizaron. Un error de
altura impar del montaje se resolvió volviendo a renderizar resultados guardados,
sin repetir seguimiento ni ajustar la configuración. Los hashes de módulos en
`*_details.json` identifican el código efectivo de cada worker; el protocolo
inicial y su nota de reanudación conservan la historia de preparación.

Validación: 85 pruebas pasan; los vídeos tienen 193 fotogramas y sus PTS coinciden
con el mapa (tolerancia 20 microsegundos), con factor cuatro en la versión lenta.
Se inspeccionaron imágenes de las tres ventanas para comprobar la presentación.

Antes de otra configuración se requiere revisión humana de la cobertura de la
bola por la región cerca de cada golpe y del jugador/fase inferidos. La siguiente
prueba de asociación alrededor del contacto o contraste local es una propuesta,
no una ejecución autorizada por este checkpoint. H2 y la ampliación a más clips
continúan pendientes.
