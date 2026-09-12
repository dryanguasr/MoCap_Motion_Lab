# Ball tracking core — Fase A

Esta rama prepara el núcleo numérico para seguimiento de la bola antes de adaptar la detección visual a videos reales.

## Alcance actual

Incluye:

- estado 3D de bola: posición, velocidad, tiempo y spin;
- propagación con gravedad;
- drag cuadrático;
- Magnus opcional, desactivado por defecto;
- integración RK4;
- observaciones visuales 2D separadas explícitamente de predicciones físicas;
- asociación de candidatos por distancia respecto a la proyección esperada y confianza visual;
- ajuste robusto de estado inicial a observaciones 3D métricas;
- configuración física externa en `config/ball_tracking_defaults.json`;
- pruebas sintéticas.

No incluye todavía un detector de bola específico para los videos de entrenamiento.

## Decisión metodológica principal

Una detección visual y una predicción física son objetos diferentes.

- `BallObservation`: evidencia visual en píxeles.
- `BallState`: estado físico 3D estimado/predicho.

El código futuro de visualización debe distinguir ambas fuentes y nunca dibujar una predicción como si fuese una observación.

## Modelo de vuelo

El estado se propaga con:

1. gravedad;
2. resistencia aerodinámica cuadrática;
3. fuerza de Magnus opcional.

La bola se representa inicialmente con masa 2.7 g y radio 20 mm. Los demás coeficientes son parámetros configurables/priores y no deben presentarse como parámetros individualmente identificados a partir de un video monocular.

### Magnus

El modelo usa el componente de spin perpendicular a la velocidad para construir un coeficiente de sustentación limitado explícitamente. El spin se mantiene constante durante un tramo corto de vuelo.

`magnus_enabled` es `false` por defecto.

Esto es deliberado: primero deben demostrarse tracking y velocidad lineal reproducibles. El spin obtenido solo a partir de trayectoria se considerará `effective spin` mientras no se demuestre identificabilidad.

## Asociación observación–predicción

`associate_observations(...)` recibe:

- estado físico predicho;
- candidatos visuales del mismo instante;
- función de proyección cámara 3D → píxel;
- umbral espacial de gating;
- escala esperada del error de proyección.

Un candidato muy alejado de la predicción física se rechaza aunque tenga alta confianza visual. Dentro del gate, la puntuación combina confianza visual y proximidad a la predicción.

Esta función no corrige todavía el estado 3D desde una sola observación monocular.

## Ajuste robusto

`fit_initial_state_world(...)` recupera posición/velocidad iniciales a partir de observaciones 3D métricas usando Gauss–Newton amortiguado y pesos Huber.

Su propósito inmediato es:

- validar numéricamente el motor con trayectorias sintéticas;
- servir posteriormente para datos multivista/calibrados;
- separar la física del problema de reconstrucción monocular.

**No es un solucionador píxel → 3D monocular.**

## Validación sintética

Ejecutar:

```powershell
python scripts/validate_ball_tracking_synthetic.py
pytest tests/test_ball_tracking.py
```

La prueba sintética genera una trayectoria conocida, añade ruido controlado y una observación degradada, y luego intenta recuperar el estado inicial.

Que esta prueba pase solo demuestra consistencia numérica interna. No valida parámetros aerodinámicos ni exactitud sobre una bola real.

## Handoff a Fase B / Codex local

Sobre videos reales, el siguiente trabajo debe ser:

1. generar candidatos visuales por frame;
2. caracterizar blobs nítidos y streaks por motion blur;
3. estimar/calibrar geometría de cámara y mesa;
4. proyectar la trayectoria física al plano de imagen;
5. asociar detecciones a predicciones;
6. mantener track durante pérdidas breves;
7. detectar bote e impacto con raqueta;
8. estimar velocidad saliente;
9. habilitar/ajustar Magnus solo si mejora resultados de forma identificable y reproducible.

La prioridad científica antes de retomar cinética corporal es relacionar variables cinemáticas del golpe con velocidad saliente de la bola y, si los datos lo permiten, con un spin efectivo inferido.
