# Prueba limitada al primer intercambio

Estado: el primer intercambio ya se completó con 38 semillas manuales. El
[diagnóstico técnico y contraste visual](../data/processed/reports/professionals__first_rally_diagnostic.md)
registra el resultado y la hipótesis de búsqueda guiada por gestos, aún sin
implementar. El lanzador sigue limitado a este clip para reproducirlo.

Antes de etiquetar más clips, terminar únicamente `rally_001_part_01` usando
`track_professionals.cmd`. Sus límites aceptados son [0,481) en el original.
El lanzador reanuda el mismo registro de intervenciones y termina después de
ese clip. No se cambian las anotaciones ni la aceptación de otros intercambios.
Una salida anticipada conserva la sesión pendiente; no significa que el clip
esté completo. El progreso de esta prueba se calcula sobre un solo intercambio.

## Diagnóstico al completar la anotación

Usar el resumen, CSV y vídeo con prefijo
`Ma-Long-and-Fan-Zhendong-Training-T2-Diamond-2019-Malaysia__rally_001_part_01__ball_tracking_assisted`
en `data/processed/summaries`, `metrics` y `videos`. Conservar también el registro
`data/annotations/rallies/interventions/rally_001_part_01.json`.

1. Revisar visualmente cada recuperación: bola correcta, mano/raqueta, fondo u
   otra bola. Separar identidad comprobada de recuperación declarada por el motor.
2. Medir duración de seguimiento correcto tras cada semilla e intervenciones
   por segundo. Separar observaciones, predicciones, omisiones y puntos manuales.
3. Localizar errores respecto a preparación del golpe, contacto, vuelo, rebote
   y final del intercambio. Comprobar los fotogramas y PTS originales.
4. Repetir la prueba que oculta la siguiente etiqueta usando solo las anteriores.
   Los puntos de intervención son casos difíciles, no una muestra aleatoria de
   exactitud. Las etiquetas por sí solas tampoco validan los fotogramas intermedios.
5. Decidir si merece la pena continuar etiquetando o mejorar primero la asociación.

## Hipótesis del usuario para una siguiente iteración — no implementada

Usar preparación y ejecución del golpe para anticipar la zona frontal del
jugador y acotar temporalmente la búsqueda. Dentro de esa región, aplicar resta
de imágenes/fondo para localizar el borrón. Con una dirección de movimiento
suficientemente respaldada, evaluar el extremo delantero del borrón como posición
semántica de la bola, en lugar del centro de la mancha.

Evaluar esta correspondencia como hipótesis, incluyendo excepciones durante
oclusiones, rebotes, cambios de dirección y final del intercambio. Evitar que
una región demasiado estrecha excluya la bola real; distinguir incertidumbre
del gesto, de la región, de la dirección y de la detección. El extremo del borrón
no es intercambiable con su centro sin definir qué instante de la exposición
representa la posición y mantener esa convención en etiquetas y mediciones.

Comparación futura en este mismo clip: sistema actual, región guiada por gesto,
región más resta local, y estimación del extremo con dirección. Usar etiquetas
reservadas para evaluación para no medir sobre los mismos puntos utilizados al
ajustar parámetros. No se ha añadido ninguna de estas variantes al seguidor.
