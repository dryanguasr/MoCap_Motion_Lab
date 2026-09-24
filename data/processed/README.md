# Organización de salidas

Todas las salidas generadas se agrupan por tipo. Los nombres siguen el patrón
`fuente__pipeline__artefacto.ext`, de modo que cada archivo conserva su origen y
su función aunque ya no esté dentro de una carpeta por video.

| Carpeta | Contenido |
|---|---|
| `videos/` | Videos anotados o diagnósticos |
| `metrics/` | Series por frame y resúmenes numéricos CSV |
| `events/` | Eventos semánticos o candidatos de contacto/bote |
| `datasets/` | Tablas preparadas para análisis posterior |
| `summaries/` | Resúmenes compactos JSON/CSV |
| `arrays/` | Cachés y series numéricas NPZ/NPY |
| `diagnostics/` | Imágenes, gráficas y hojas de contacto |
| `reports/` | Informes Markdown |
| `metadata/` | Manifiestos, supuestos y metadatos de ejecución |
| `logs/` | Logs de ejecución cuando se generen |

Los videos fuente permanecen separados en `data/raw/`. Los artefactos pesados
siguen excluidos de Git salvo que ya estuvieran versionados históricamente.
