# Reparación de tomas y reutilización de geometría

El manifiesto usado durante la anotación agrupaba los fotogramas 526–10470
como una sola toma, aunque hay un corte al plano cercano en 1216. También
agrupaba 10471–11615, con un corte de vuelta al plano cercano en 11326 y
un último fotograma negro en 11615. Una única geometría no podía satisfacer
las comprobaciones de inicio, centro y final; cada corrección deshacía otra.

La inspección secuencial del original identificó las tomas de juego:

| Toma | Fotogramas incluidos | Geometría reutilizada |
|---|---|---|
| General A | 0–1215 | Anotación del usuario confirmada para shot_001 |
| Cercana A | 1216–10470 | Borrador del usuario para shot_73352fca |
| General B | 10471–11325 | Misma anotación del plano general |
| Cercana B | 11326–11614 | Mismo borrador del plano cercano |

La frontera 526 no cambia el encuadre y se retiró. Desde 11615 queda el
tramo de salida sin intercambios aceptados y sin necesidad de geometría.
Se preservaron exactamente los identificadores, límites y aceptación de los
23 intercambios del usuario. Todos caben en una única toma corregida.

Las coordenadas existentes se copiaron sin introducir clics nuevos ni transformar
la imagen. El asistente examinó recortes ampliados con ambas superposiciones
en inicio, centro y final de cada toma: doce imágenes en total. Se conservan
en `data/processed/diagnostics/geometry_repair`. Es una comprobación visual
aproximada, no una medición de precisión ni una nueva confirmación humana.

Los archivos inferidos registran el hash y la ruta de la anotación o borrador
de origen, la familia de cámara y el estado `inferred_assistant_visual_checked`.
Los originales se conservan. Una copia adicional del manifiesto y las anotaciones
previas está en `data/annotations/rallies/backups/geometry_repair_20260909_193046`.

El manifiesto reparado pasa validación de fuente, cachés, geometrías y partes.
El revisor ya no tiene decisiones pendientes para estos intercambios. El
procesador puede trabajar directamente sobre las vistas del original; abrir
el revisor también permite exportar los clips físicos. No se ha ejecutado aquí
el seguimiento asistido ni se han exportado los 23 MP4.

Para nuevas propuestas se añadió una señal de cambio espacial que también
detecta cortes entre encuadres con colores parecidos cuando falla la asociación
de características. Sigue siendo una propuesta que requiere revisión.
