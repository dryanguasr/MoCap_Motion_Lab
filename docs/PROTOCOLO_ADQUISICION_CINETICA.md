# Protocolo propuesto: cinética del jugador de tenis de mesa

Versión 1 · 2026-09-02 · Sujeto de referencia: talla declarada 1.84 m, masa aproximada 90 kg.

Objetivo: obtener fuerzas de reacción del suelo por pie y momentos articulares netos tridimensionales reproducibles. La prioridad de seguimiento sigue siendo el jugador principal y su muñeca derecha; ambos miembros inferiores deben permanecer visibles. Este documento es un diseño experimental, no una validación ni una prescripción clínica.

## 1. Qué necesitamos medir y qué seguirá siendo estimado

La dinámica inversa necesita cinemática, parámetros inerciales y todas las cargas externas relevantes, medidas o modeladas. Su resultado son momentos netos: no identifica por sí sola fuerzas musculares individuales, cocontracción o presión dentro de la articulación. Véase [documentación de dinámica inversa de OpenSim](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090063).

| Variable | Captura ideal | Limitación que resuelve |
|---|---|---|
| Movimiento corporal 3D | 3–4 cámaras sincronizadas y calibradas; referencia óptica con marcadores en un subconjunto | Profundidad, perspectiva, oclusión, centros articulares |
| Fuerza, centro de presión y momento libre de cada pie | Plataformas de fuerza multiaxiales independientes, integradas al suelo | Reparto entre pies y punto de aplicación, indeterminados con video solo |
| Masa corporal y talla | Báscula contrastada con masas patrón y estadiómetro | Escala real de fuerzas y geometría |
| Masa, centro de masa e inercia de segmentos | Modelo antropométrico personalizado y análisis de incertidumbre | La talla y el peso no determinan estos parámetros individualmente |
| Movimiento/carga de raqueta | Raqueta pesada, centro de masa e inercia caracterizados; marcadores o IMU sincronizada | La muñeca no es el centro de la raqueta ni mide su rotación |

La prioridad instrumental para mejorar los torques es obtener contactos medidos y cinemática 3D. Un impedómetro complementa esta adquisición, pero no sustituye esas mediciones. La decisión de compra/alquiler debe tomarse después de confirmar disponibilidad local y compatibilidad de sincronización, no a partir de una marca concreta.

## 2. Papel correcto del InBody u otro impedómetro

La salida segmentaria habitual se refiere a cinco regiones amplias: ambos brazos, ambas piernas y tronco. La masa libre de grasa no equivale a masa muscular ni a masa total del segmento. Las variables disponibles dependen del modelo. No sumar agua o minerales nuevamente a una masa libre de grasa que ya los incluye. Fuente: [guía profesional del fabricante](https://uk.inbody.com/wp-content/uploads/2018/08/The_Professional_s_Guide_to_the_InBody_Result_Sheet.pdf).

**Consecuencia para nuestro modelo:** no podemos asignar directamente el número de una pierna a muslo, pantorrilla y pie por separado. Tampoco obtenemos centros de masa ni tensores de inercia con esos números. Que una estimación sea individualizada no demuestra que sea más exacta.

Propuesta de uso:

1. Conservar el informe original, unidades, definición de cada variable y límites regionales. Registrar marca, modelo, número de serie, versión de software, fecha y condiciones de la sesión. Exportar impedancias crudas y frecuencias si el equipo lo permite.
2. Separar datos medidos por otros instrumentos —masa de báscula y longitudes— de estimaciones de composición. No tratar porcentajes respecto a una población de referencia como kilogramos.
3. Construir un modelo previo de segmentos rígidos. Ajustar sus masas con restricciones de positividad y suma igual a la masa corporal. Solo incorporar totales regionales cuando la definición del equipo sea compatible; usar masa magra aislada como información parcial, no como masa total.
4. Mantener explícita la incertidumbre al subdividir una región. La división de una pierna en tres enlaces depende de longitudes, perímetros/geometría y supuestos de distribución, aunque su total regional esté estimado.
5. Evaluar el modelo antropométrico sin BIA frente al modelo con BIA en los mismos ensayos de referencia. Aceptar la complejidad adicional solo si mejora resultados fuera de la muestra de ajuste.

Si el estudio justifica mayor caracterización, puede valorarse DXA con regiones definidas o información volumétrica adicional, con personal competente y autorizaciones correspondientes. No se presupone necesaria una exploración radiológica para este piloto, ni que proporcione por sí sola todos los parámetros inerciales.

### Condiciones de BIA

Realizarla antes del entrenamiento, en condiciones repetibles y siguiendo el manual del modelo. El fabricante recomienda hidratación habitual, misma franja horaria, permanecer de pie 5–10 minutos y esperar al menos tres horas tras una comida; no medir inmediatamente después de ejercicio. Registrar desviaciones y no provocar deshidratación. El operador debe comprobar las precauciones y contraindicaciones del dispositivo. Fuente: [preparación para InBody](https://inbodyusa.com/general/inbody-test/).

En la primera sesión propongo dos mediciones con reposicionamiento, si el manual lo permite, para cuantificar repetibilidad técnica; discrepancias se investigan, no se promedian sin explicación. Registrar entrenamiento previo, hora de comida/bebida, condiciones ambientales y cualquier factor que el operador considere relevante, sin interpretar clínicamente los resultados.

## 3. Antropometría y caracterización del material

- Medir masa corporal con ropa ligera y sin raqueta. Registrar por separado calzado, ropa relevante y equipos añadidos para evitar doble conteo. Repetir talla y masa; no mantener 90 kg como constante si existe medición actual.
- Medir longitudes bilaterales entre referencias anatómicas documentadas: brazo, antebrazo, mano, muslo, pierna y pie; ancho de pelvis y hombros. Registrar el examinador y repetir una parte para estimar error de colocación.
- Usar pose estática para ajustar geometría, no asumir que el torso mide exactamente 28.8 % de la talla. Separar longitud del segmento, posición de su centro de masa e inercia.
- Pesar la raqueta completa con sus gomas y accesorios; localizar su centro de masa por equilibrio. Si la dinámica de muñeca/raqueta es un objetivo, medir o estimar su inercia con un procedimiento documentado y controlado; una masa puntual en la muñeca no basta.
- Conservar los parámetros poblacionales empleados y comprobar si la población de referencia es adecuada. [De Leva, 1996](https://pubmed.ncbi.nlm.nih.gov/8872282/) describe ajustes antropométricos de referencia, no mediciones individualizadas del jugador.

## 4. Disposición y calibración de cámaras

Propuesta inicial de ingeniería, a verificar en una sesión piloto:

- Tres o cuatro vistas convergentes, al menos dos vistas útiles por articulación, sin que mesa/oponente oculten simultáneamente pies, pelvis y muñeca derecha. Incluir todo el volumen de desplazamiento y margen para la raqueta. Cámaras fijas: no zoom ni estabilización digital variable.
- Capturar a 120–240 fps si la iluminación y resolución permiten localizar las articulaciones. Ensayar exposición corta, inicialmente alrededor de 1/1000 s, y comprobar desenfoque, parpadeo y ruido antes de fijarla. No son requisitos universales ni garantizan resolver el impacto pelota–raqueta.
- Calibración intrínseca por cámara con patrón de dimensiones verificadas, cubriendo encuadre y orientaciones. Extrínseca compartida con patrón/varilla visible en varias vistas y en diferentes posiciones y alturas del volumen de trabajo.
- Definir un sistema global diestro: Z vertical real, X según una dirección marcada en el suelo, Y completando el sistema. Medir la transformación de cada plataforma a ese sistema. La superficie de la mesa no es el suelo.
- Verificar escala con distancias independientes NO usadas en el ajuste. Reservar capturas del patrón para comprobar error de reproyección y consistencia 3D.
- Recalibrar si se mueve una cámara, cambia óptica/resolución o falla la comprobación final. Guardar calibraciones originales y derivadas con fecha e identificador.

Como objetivos iniciales de control propongo mediana de reproyección <1 px y P95 <2 px, y error métrico de referencias independientes <1 %. Deben adaptarse al equipo y al error cinético tolerable; superar estos objetivos no demuestra exactitud articular.

## 5. Plataformas y sincronización

Para movimientos laterales, usar una superficie instrumentada suficiente para la trayectoria natural, no obligar al jugador a acertar una plataforma pequeña. Cada pie debe apoyarse en plataformas identificables por separado. Registrar apoyos sobre bordes o sobre dos sensores; si no se pueden resolver, excluir el ensayo.

Propuesta: fuerzas a 1000 Hz o más, con filtrado antialias apropiado y calibración del fabricante vigente. Antes de la sesión: puesta a cero sin carga, comprobación con cargas conocidas, comprobación del centro de presión en posiciones conocidas y transformación espacial al sistema de cámaras. Registrar también el momento libre, no solo Fz.

Usar disparo/reloj compartido cuando sea posible. Como alternativa, un evento luminoso capturado por todas las cámaras y registrado en la adquisición de fuerzas, al principio y al final para medir desfase y deriva. Guardar timestamps y verificar pérdidas de frames. Objetivo inicial: desfase residual <2 ms si el hardware lo permite; cuantificar su efecto sobre torques mediante desplazamientos temporales de sensibilidad. No asumir que audio, fps nominales o pulsar «grabar» simultáneamente sincronizan los sistemas.

Las plantillas de presión pueden complementar el contacto y su distribución normal; no se deben tratar como sustituto de la fuerza tridimensional y el momento libre sin demostrar esas capacidades del sistema concreto.

## 6. Secuencia de una sesión

1. Consentimiento, identificador seudónimo, criterios de seguridad y revisión del estado del material por personal responsable.
2. BIA y antropometría antes de esfuerzo; documentar las condiciones anteriores.
3. Calibraciones espaciales, cero/cargas conocidas de plataformas y sincronización.
4. Registro estático A/T y de pie relajado, unos 5 s por condición. Comprobar que la suma de Fz en reposo corresponde al peso del sistema realmente instrumentado.
5. Movimientos funcionales suaves para ajustar centros/ejes articulares y comprobar reconstrucción; calentamiento deportivo habitual supervisado.
6. Bloques cortos por tarea: posición de espera, transferencia lateral, golpe controlado, secuencia deportiva. Objetivo inicial: al menos cinco ensayos técnicamente utilizables por condición; 5–10 repeticiones por bloque según tolerancia y tarea. Registrar orden, intensidad, descansos y fatiga; no imponer velocidad o volumen inseguro.
7. Incluir ensayos sin pelota para separar cinemática de golpe y perturbación de impacto. Con pelota, medir/modelar la carga externa si es un objetivo; de lo contrario, marcar ventanas de impacto como no evaluables para esa pregunta.
8. Repetir evento de sincronización y control de escala/cero al cierre. Anotar contactos con mesa, oclusiones, salidas del volumen y cualquier incidencia.

Se propone inicialmente una discrepancia de fuerza estática <3 % del peso medido como alarma de montaje, no como certificación. Guardar los datos rechazados y su motivo; no repetir solo los ensayos que producen valores «bonitos».

## 7. Procesamiento y validación antes de interpretar

Reconstruir pose 3D multivista y ajustar un modelo con longitudes constantes y articulaciones coherentes. Mantener visibles residuos de reproyección, pérdidas y ajustes. No convertir la confianza de un detector en una probabilidad de exactitud cinética.

Elegir el filtrado de cinemática y fuerzas mediante análisis de ruido y sensibilidad temporal/frecuencial; documentar fases y retardos. La segunda derivada amplifica errores. No rellenar huecos largos o impactos con interpolación y presentarlos como observaciones.

Introducir cargas por pie, puntos de aplicación, momento libre y raqueta en el modelo personalizado. Calcular momentos netos 3D y reportarlos en N·m y N·m/kg, con ejes/signos anatómicos explícitos. Revisar residuos de fuerza/momento globales; un residuo pequeño obtenido por construcción no valida los datos.

Comparar el método de video con plataformas y, en un subconjunto ideal, cinemática óptica con marcadores. Para torques, la referencia también es una estimación de dinámica inversa, basada en esos instrumentos, no una lectura directa de un sensor articular.

Reportar MAE/RMSE, sesgo, error de picos y de tiempos de eventos, resultados por eje/articulación, cobertura y repetibilidad entre sesiones. Separar ajuste de validación por ensayo/sesión; recortes del mismo video no son muestras independientes. Prerregistrar tolerancias tras el piloto según el uso previsto.

Existen sistemas como [OpenCap](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462) que combinan video multivista y simulación para estimar dinámica. Esa publicación no valida automáticamente este montaje monocular ni el tenis de mesa: haría falta validación específica de los gestos, velocidades y oclusiones presentes.

## 8. Entregables y trazabilidad

Conservar videos, plataformas y reportes de instrumentos como originales inmutables con SHA-256. Vincular sujeto, sesión, ensayo, cámaras, calibración y versión de análisis mediante un manifiesto. Guardar unidades, ejes, timestamps, frecuencia real, parámetros antropométricos, filtros, contactos y motivos de exclusión. No subir datos corporales identificables al repositorio.

Orden de mejora recomendado: (1) sincronización + 3D + fuerzas por pie; (2) antropometría personalizada, con BIA complementaria; (3) caracterización completa de raqueta e impactos; (4) validación independiente antes de usar resultados para decisiones sobre carga, técnica o lesión.

## 9. Relación con el piloto actual

El piloto usa únicamente seis clips cortos ya disponibles, 90 kg, 1.84 m y un modelo planar con supuestos declarados. No se dispone todavía de BIA, plataformas, calibración 3D ni medidas reales de raqueta. Sus resultados no cumplen este protocolo final. Ver el informe generado en `data/processed/reports/batch__planar_kinetics__report.md` y las hipótesis reproducibles en `scripts/estimate_planar_kinetics.py`.
