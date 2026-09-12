# Alcance de la extrapolación desde las correcciones guardadas

Se revisaron 32 semillas visibles del primer intercambio, entre los fotogramas
17 y 432. No se modificaron el registro de intervenciones ni los resultados
asistidos guardados. Las etiquetas se utilizaron como referencia aproximada.

## Comprobación con etiquetas ocultas

Para cada una de las 31 semillas posteriores a la primera se reprodujo el
seguidor utilizando únicamente las semillas anteriores. Se comparó su resultado
en el fotograma de la etiqueta oculta con el punto señalado por el usuario.

- En 30 casos no produjo una observación visual en ese fotograma.
- En el único caso con observación, el error fue de unos 768 píxeles.
- Ninguno de los 31 casos quedó dentro de 35 píxeles de la etiqueta oculta.

Estos son precisamente puntos difíciles elegidos para intervenir, por lo que
no representan la exactitud de todos los fotogramas. Sí muestran que el método
actual no permite prescindir de esas correcciones. «Recuperación confirmada»
significa que se encontraron dos detecciones coherentes, no que se comprobó la
identidad de la bola. También se inspeccionaron doce imágenes posteriores a
semillas con la posición automática superpuesta; no se realizó una anotación
exhaustiva de identidad en todos los fotogramas.

La extrapolación de velocidad constante usando dos etiquetas para predecir la
siguiente tuvo un error mediano de 440 píxeles en 30 ternas. Solo tres ternas
tenían ambos intervalos temporales de hasta 0,15 s; tampoco respaldaron ese
modelo (mediana 397 píxeles). Golpes y rebotes invalidan prolongar una dirección
sin nuevas observaciones. No se generaron etiquetas automáticas a partir de
esa extrapolación.

Los puntos guardados sirven para reiniciar localmente y para evaluar futuras
mejoras de asociación visual. No hay evidencia suficiente para completar
automáticamente los demás intercambios o rellenar todos los huecos actuales.

## Interfaz y progreso

Se corrigió la línea verde para que no una observaciones a través de pérdidas
ni reinicios manuales. Esto elimina segmentos dibujados engañosos, pero no
corrige por sí solo las detecciones equivocadas del seguidor.

La interfaz muestra avance del clip y global, duración de vídeo restante y
número de intercambios posteriores. Se calcula con PTS reales y el comienzo del
episodio pendiente, independientemente de la navegación. El porcentaje mide
recorrido procesado y no exactitud de seguimiento.

En la sesión examinada: 91,5 % del primer intercambio, 4,6 % del conjunto;
0,87 s pendientes en ese clip y 192,68 s de vídeo en total, incluidos otros
22 intercambios. No es una estimación de duración del trabajo humano.

Resultados detallados y hoja visual: `data/processed/diagnostics/user_seed_audit/`.
Validación de código: 80 pruebas pasan, incluyendo cortes de trayectoria y
progreso VFR; la reanudación se comprobó de nuevo tras ajustar la persistencia.
