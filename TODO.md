# ✅ Lista de Tareas Pendientes (`TODO`)

Registro operativo de tareas pendientes de desarrollo experimental, integración de fuentes abiertas y publicación del proyecto en abierto.

## 📊 Hoja de Trabajo Técnica (Fases 3 a 5)
- [x] **Cierre de Capa Estructural (`notebooks/09_EDA_maestro.ipynb`):**
  - Realizado análisis exploratorio sobre `maestro_gtfs.parquet` (2.232.015 tramos unificados).
  - Implementado *Feature Engineering* relacional estructural (`tipo_jornada`, `indice_sinuosidad_normalizado`, `velocidad_teorica_kmh`, `saturacion_marquesina_15min`, `friccion_acumulada_m30`, `hora_seno` y `hora_coseno`).
  - Llevadas las 9 variables a producción en `src/consolidacion.py`, documentadas en `docs/09_maestro_Dict.md` y validadas con `pytest` (`6 passed`).
- [ ] **Modelo 1 — Baseline Estructural (`GTFS + API MobilityLabs EMT`):**
  - Implementar captura de tiempos reales de llegada en parada y cálculo de la variable objetivo (`retraso_real_seg`).
  - Entrenar y evaluar el **Modelo 1** usando exclusivamente variables planificadas GTFS, dejando fijadas las métricas base (`MAE`, `RMSE`, `R²`) y el conjunto fijo de validación (*test set*).
- [ ] **Modelo 2 — Enriquecimiento con Tráfico Urbano (`notebooks/10_EDA_trafico.ipynb`):**
  - Descargar y auditar datos abiertos del **estado del tráfico en tiempo real del Ayuntamiento de Madrid** (intensidad, ocupación, carga y nivel de servicio por punto de medida `lat/lon`).
  - Ejecutar cruce espacial por proximidad con las paradas EMT y entrenar el **Modelo 2**, midiendo la mejora respecto al Modelo 1.
- [ ] **Modelo 3 — Enriquecimiento Meteorológico y Comparativa Final (`notebooks/11_EDA_meteorologia.ipynb` y `12_Comparativa_Modelos.ipynb`):**
  - Integrar datos meteorológicos horarios (precipitación, rachas de viento, temperatura) por ventana temporal.
  - Entrenar el **Modelo 3** y construir el estudio comparativo final enfrentando los 3 modelos e interpretando la importancia de variables (*SHAP / Feature Importance*).

## 🚀 Checklist de Publicación (Paso de Privado a Público en GitHub)
- [x] **Cambiar visibilidad en GitHub:** Repositorio publicado en abierto (`Public`).
- [x] **Activar GitHub Pages:** Sitio web desplegado en `https://github.io`.
- [x] **Actualizar enlaces interactivos en los Notebooks (`GitHub Pages`):**
  - En `notebooks/02_EDA_stops.ipynb`: Enlace activo a `https://github.ioreports/mapa_paradas_emt.html`.
  - En `notebooks/03_EDA_shapes.ipynb`: Enlace activo con selector de rutas y leyenda reubicada en `https://github.ioreports/visor_trazado_emt.html`.
- [x] **Auditoría visual final en GitHub:** Confirmado que los notebooks renderizan todos sus gráficos estáticos sin espacios en blanco.
