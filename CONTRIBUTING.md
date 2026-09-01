# Contribución

## Flujo recomendado

1. Crear una rama por tarea o experimento.
2. Mantener los cambios pequeños y documentados.
3. No subir videos de participantes ni datasets experimentales sin una decisión explícita del equipo.
4. Separar cambios de infraestructura común de aplicaciones particulares.
5. Incluir en el README o documentación de cada experimento:
   - objetivo;
   - datos utilizados;
   - versión del modelo/librería;
   - parámetros de procesamiento;
   - limitaciones conocidas;
   - criterio de evaluación.

## Convención de ramas sugerida

- `feature/...`
- `experiment/...`
- `fix/...`
- `docs/...`

## Criterio para incorporar una nueva librería de pose

No reemplazar MediaPipe únicamente porque otra herramienta produzca una visualización más atractiva. Comparar al menos:

- precisión o error frente a una referencia;
- estabilidad temporal;
- velocidad de procesamiento;
- requisitos de hardware;
- facilidad de despliegue;
- licencia;
- mantenimiento del proyecto;
- disponibilidad de landmarks relevantes para la aplicación.
