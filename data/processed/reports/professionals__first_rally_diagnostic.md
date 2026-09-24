# Primer intercambio: diagnóstico técnico para contrastar con inspección visual

Los artefactos citados por nombre en este informe están en la carpeta local
`data/processed/first_rally_review/`; los vídeos, métricas por fotograma y copias
de anotaciones permanecen fuera de Git según la política de datos del proyecto.

## Material revisado

Versión final con 38 semillas visibles, 481 fotogramas (original 0–480), unos
10,233 s de vídeo. Se reprodujeron las intervenciones guardadas en una copia
separada del registro. El vídeo `primer_intercambio_anotado.mp4` conserva
2156×1214 y todos los PTS relativos del original, verificados con FFprobe.
No contiene audio. No se procesaron otros intercambios.

La cruz rosa es una semilla manual; verde significa una detección seleccionada,
no una identidad verificada; naranja significa predicción. Las líneas de mesa
y malla son la geometría reutilizada y validada visualmente en la revisión previa.

## Mediciones de esta ejecución

| Resultado | Valor | Interpretación |
|---|---:|---|
| Semillas manuales | 38 | 3,71 intervenciones por segundo de vídeo |
| Mediana entre semillas | 0,183 s | Carga de intervención alta |
| Recuperaciones declaradas | 38 | Dos detecciones compatibles; no prueba de identidad |
| Mediana hasta confirmar recuperación | 0,050 s | Reinicializa rápido, sin garantizar continuidad |
| Fotogramas OBSERVED | 359/481 (74,6 %) | Detecciones seleccionadas, mayoritariamente asistidas |
| Fotogramas PREDICTED | 15 | Predicciones, separadas de observaciones |
| Fotogramas LOST | 101 | Incluyen 38 semillas, 7 huecos anteriores a una semilla y 56 restantes |
| Fotogramas UNINITIALIZED | 6 | Inicio sin trayectoria |
| Tramos consecutivos OBSERVED, sin cruzar reinicios | 54 | Continuidad declarada por el motor |
| Mediana / máximo de esos tramos | 0,10 / 0,65 s | No equivalen a duración de seguimiento correcto |

Al ponderar por la duración real de cada fotograma: 7,709 s con observación,
0,245 s de predicción, 2,145 s perdidos y 0,134 s sin inicializar. Los 101 LOST
no deben describirse como 101 pérdidas espontáneas: los clics se representan
deliberadamente como semillas y nunca como detecciones automáticas.

Fuentes seleccionadas: 240 manchas visuales del modo de recuperación manual,
20 combinaciones de ese modo con contraste temporal, 60 combinaciones con
RacketVision, 38 detecciones de RacketVision y una de contraste temporal ordinario.
La cifra histórica `automatic_observation_frames` del resumen cuenta detecciones
del algoritmo, aunque estén condicionadas por asistencia; no mide cobertura autónoma.

## Perspectiva técnica

1. **La reinicialización no es el principal indicador de éxito.** El motor confirma
   las 38 semillas, pero pierde continuidad rápidamente. En 1,03; 5,63; 6,92;
   9,07 y 9,52 s las respectivas semillas solo sustentan dos fotogramas OBSERVED
   antes de la siguiente intervención. Son zonas prioritarias para contraste visual.
2. **El espacio de candidatos es muy permisivo.** El radio de recuperación visible
   es `35 + 6000·dt` píxeles: 335 px a los 0,05 s y 1535 px a los 0,25 s. El
   borrón permitido llega a 260 px de longitud y 6000 px² de área en este vídeo.
   Esto aumenta la sensibilidad, pero admite manos, ropa, raquetas y otros brillos.
   Que estos parámetros causen cada error concreto requiere comprobar el vídeo.
3. **La coherencia actual no prueba identidad.** Se aceptan dos detecciones futuras
   con velocidad 70–6000 px/s, separación máxima de 0,14 s y extrapolación a la
   semilla dentro de 35 px (90 px si fuera estimada). La confirmación expira a
   los 0,25 s. Una pareja de distractores también puede satisfacer estas condiciones.
4. **La asociación cerca del golpe es una zona delicada.** La fusión da peso a la
   proximidad a la raqueta; a menos de 130 px suspende la penalización corporal.
   Durante eventos la puerta de asociación se multiplica por 2,2 respecto a una
   base de 75 px, ampliada 28 px por fotograma perdido. Tiene sentido para cambios
   bruscos, pero puede favorecer distractores precisamente junto a la mano.
5. **La resta ya existe, pero no el sector guiado por el gesto.** El modo de
   recuperación combina diferencia con vecinos, primer plano positivo respecto
   al fondo, brillo >=115 y saturación <=115. Opera dentro de una región amplia
   derivada de la mesa. Después de una semilla sigue aportando candidatos durante
   el resto del clip; no se limita a los primeros fotogramas de recuperación.
6. **El borrón se representa por el centro de su caja.** El código usa
   `(x+w/2, y+h/2)`, no un extremo orientado. Un desplazamiento constante respecto
   a tus clics podría deberse a esta convención y no solo a una identidad errónea.
   Es necesario contrastar qué punto del borrón estás señalando.
7. **La física es aproximada.** La imagen se proyecta en un plano vertical con
   escala derivada de la mesa, no en 3D calibrado. Usa gravedad 9,81, arrastre 0,47
   y Magnus desactivado. No conviene ajustar estos valores para compensar errores
   de detección o asignación de identidad. Las predicciones duran 4 fotogramas,
   ampliables a 6 durante eventos; con VFR no representan una duración fija.

El anclaje clásico anterior queda deshabilitado tras la semilla manual. Por
tanto, un desvío posterior puede proceder de una nueva asociación equivocada,
sin que necesariamente se esté restaurando la trayectoria vieja.

## Contraste con tu inspección

Anotar segundo o fotograma y distinguir: (a) bola correcta, (b) otra entidad,
(c) bola correcta pero posición desplazada dentro del borrón, (d) ausencia de
marcador pese a bola visible, (e) oclusión real. Revisar especialmente el cambio
de dirección junto a cada jugador y los cinco momentos señalados arriba.

Mi conclusión provisional es que la carga manual y la fragmentación todavía
son demasiado elevadas para ampliar la anotación a otros clips. La exactitud
visual y la causa dominante quedan pendientes de tu inspección. No se han
entrenado modelos ni cambiado parámetros del seguimiento en esta ejecución.

La hipótesis de acotar por preparación/ejecución merece una comparación controlada
en este clip: región guiada, resta local y, después, extremo orientado del borrón.
Debe reservarse una parte de las anotaciones para evaluar, y permitir ampliar la
región cuando la fase o dirección sean inciertas. No se ha implementado.

## Reproducibilidad y estado de exportación

### Contraste posterior con la inspección del usuario

El usuario informa que el seguimiento suele funcionar únicamente cuando la
bola viaja desde Fan Zhendong hacia Ma Long, después de cruzar la malla. En su
inspección se pierde en todos los impactos. Es una valoración visual del clip;
todavía no se han enumerado ni etiquetado individualmente los impactos para
calcular una tasa de fallo independiente.

Esta observación coincide con la fragilidad de continuidad medida y localiza
una prioridad: la transición alrededor del contacto. Añade una asimetría de
dirección que las métricas agregadas no mostraban. El diagnóstico no permite
atribuirla exclusivamente al gesto: también deben contrastarse visibilidad,
borrón, perspectiva, contraste y asociación cerca de mano/raqueta.

La siguiente prueba propuesta, aún no implementada, debe separar ambos sentidos
de vuelo y tres fases: aproximación/preparación, contacto y salida/ejecución.
Usaría el gesto para proponer jugador y región frontal; alrededor del contacto
no prolongaría ciegamente la velocidad de entrada, sino que evaluaría nuevas
hipótesis de salida con evidencia visual local. El extremo orientado del borrón
sería una variante posterior, con convención temporal y dirección explícitas.

Comparar sobre este mismo intercambio, reservando impactos para evaluación:
identidad correcta antes y después del contacto, tiempo hasta recuperar, falsas
asociaciones, cobertura de la bola real por la región propuesta e intervenciones
necesarias. Comparar sistema actual, región guiada por fase y región con resta
local; evaluar por separado el cambio de centro a extremo del borrón. No ampliar
la anotación de otros clips antes de resolver esta prueba.

`technical_metrics.json` contiene las mediciones y el desglose de cada semilla;
`interventions/rally_001_part_01.json` es la copia utilizada. Se conservan CSV y
eventos de esta reproducción. El estado completo se verificó por cobertura de
481 fotogramas y ausencia de intervención pendiente en el registro final.

La escritura inicial del resumen final encontró un límite de longitud de ruta
de Windows después de haber exportado y verificado el vídeo. Se acortó el nombre
temporal de la escritura atómica y se generó un resumen técnico con nombre corto
a partir de los CSV exportados. No se alteró el cálculo de seguimiento. Las
80 pruebas existentes pasan tras esa corrección.
