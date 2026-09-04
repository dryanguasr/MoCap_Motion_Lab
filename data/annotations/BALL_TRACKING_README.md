# Anotación manual mínima de bola

Estas referencias se marcaron inspeccionando los fotogramas originales a
resolución completa. El centro del candidato del detector se usó como punto
inicial y se aceptó o corrigió visualmente; por ello no es una anotación
independiente para evaluar precisión subpíxel, aunque sí permite auditar
identidad, cobertura, predicciones y falsos positivos. Son aproximadas: el
centro de un *streak* no siempre es el centro instantáneo de la bola durante la
exposición. `uncertainty_px` expresa esa tolerancia visual, no un intervalo
estadístico.

La anotación cubre dos tramos del clip fácil. El primero contiene un bote y un
contacto con la raqueta del jugador principal; el segundo incluye un frame que
el tracker debe predecir y un contacto con el oponente. No pretende medir
precisión sobre todo el video ni se usa como verdad física 3D.
