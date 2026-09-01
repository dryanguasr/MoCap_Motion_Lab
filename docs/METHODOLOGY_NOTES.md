# Notas metodológicas para investigación

Este documento define precauciones mínimas para evitar que un prototipo de visión sea presentado como un sistema de medición ya validado.

## 1. Naturaleza de las coordenadas

MediaPipe Pose Landmarker produce landmarks 3D de mundo estimados por el modelo. Son una salida valiosa para reconstrucción relativa del movimiento, pero su precisión debe caracterizarse experimentalmente antes de extraer conclusiones cuantitativas fuertes.

## 2. Velocidad

La derivación numérica amplifica ruido. Por eso la versión inicial:

1. filtra posiciones;
2. calcula la diferencia temporal usando timestamps reales;
3. filtra nuevamente la velocidad.

A futuro deben compararse alternativas con criterios cuantitativos, no solo por apariencia visual.

Métricas posibles:

- RMSE respecto a una trayectoria de referencia;
- error máximo;
- retardo introducido;
- relación señal/ruido;
- repetibilidad entre ensayos;
- sensibilidad a FPS, iluminación y oclusión.

## 3. Ángulos articulares

Los ángulos actuales se construyen con tres puntos. Ejemplo:

```text
hombro ─ codo ─ muñeca
```

El ángulo en el codo es un descriptor geométrico interpretable. Sin embargo, una articulación como hombro o cadera requiere sistemas de referencia segmentales si se desea obtener una descripción biomecánica completa en varios grados de libertad.

## 4. Diseño experimental mínimo para validar la plataforma

Una ruta razonable para una primera validación podría incluir:

### Ensayo A — condición estática

- sujeto inmóvil durante 20–30 s;
- evaluar dispersión de cada landmark;
- comparar diferentes distancias a cámara.

### Ensayo B — movimiento simple y repetible

- flexo-extensión de codo;
- movimiento aproximadamente planar;
- comparar repeticiones y velocidad de ejecución.

### Ensayo C — referencia geométrica

- incluir una longitud o trayectoria conocida en escena;
- estudiar error de escala y profundidad.

### Ensayo D — movimiento específico de aplicación

Solo después de caracterizar el sistema básico conviene pasar a gestos deportivos, agrícolas o funcionales más complejos.

## 5. Variables que deben controlarse o registrarse

- resolución de video;
- FPS nominal y efectivo;
- modelo de cámara;
- distancia cámara-sujeto;
- altura y orientación de cámara;
- iluminación;
- vestuario y contraste;
- presencia de oclusiones;
- lado dominante;
- actividad realizada;
- versión del software y modelo de pose.

## 6. Privacidad y ética

El proyecto procesa video de personas. Para ensayos de investigación deben definirse desde temprano:

- consentimiento informado cuando corresponda;
- política de almacenamiento de video;
- anonimización de datos derivados;
- acceso a grabaciones;
- periodo de conservación;
- finalidad del tratamiento de datos.

El repositorio no versiona videos ni sesiones experimentales por defecto.

## 7. Particularidades del tenis de mesa

El tenis de mesa impone retos adicionales:

- velocidades altas de brazo y muñeca;
- oclusión por tronco, pala y mesa;
- cambios rápidos de orientación;
- movimientos tridimensionales;
- necesidad potencial de mayor FPS;
- interés en orientación de mano/pala, no solo posición de muñeca.

Por esto, el módulo general debe madurar antes de asumir que una webcam convencional es suficiente para cuantificar golpes con precisión.

## 8. Preguntas de investigación posibles

La plataforma común permite que diferentes estudiantes investiguen preguntas distintas, por ejemplo:

- ¿qué filtro ofrece el mejor compromiso entre ruido y retardo?
- ¿cómo cambia el error con distancia y orientación de cámara?
- ¿qué landmarks son más confiables para una tarea específica?
- ¿puede una configuración monocular identificar fases de un gesto?
- ¿cuánto mejora una solución multicámara?
- ¿qué tan bien se correlacionan las variables estimadas con un sistema de referencia?
