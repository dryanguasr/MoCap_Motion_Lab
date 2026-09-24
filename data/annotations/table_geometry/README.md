# Anotación fija de mesa y malla

Cada toma requiere ocho puntos sobre un fondo mediano temporal:

1. superficie: `far_left`, `far_right`, `near_right`, `near_left`;
2. malla: `top_left`, `top_right`, `base_right`, `base_left`.

Ejecute:

```powershell
.\.venv\Scripts\python.exe scripts\annotate_table_geometry.py data\raw\CLIP.mp4
```

Use clic izquierdo para un punto visible y `Shift` + clic para uno ocluido que
deba estimarse por continuidad. `U` o clic derecho deshace, `R` reinicia, `S`
guarda y `Q` sale. El JSON incluye el SHA-256 del video para impedir que una
calibración se aplique accidentalmente a otra toma.
