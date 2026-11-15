# ✅ Lista de Tareas Pendientes (`TODO`)

Registro operativo de tareas pendientes de desarrollo, integración de datos y publicación del proyecto en abierto.

## 📊 Próximos Pasos de Desarrollo (Fases 2 y 3)
- [ ] **Cerrar Capa Estática (`notebooks/09_EDA_maestro.ipynb`):**
  - Realizar análisis exploratorio sobre `maestro_gtfs.parquet` (2.232.015 tramos unificados).
  - Implementar *Feature Engineering* relacional estructural (`velocidad_teorica_kmh`, presión de autobuses por marquesina/hora, fricción acumulada interior M-30 e indicadores horarios cíclicos).
  - Llevar las nuevas variables validadas a producción en `src/consolidacion.py`, documentar en `docs/09_maestro_Dict.md` y actualizar `tests/test_pipeline.py`.
- [ ] **Integración de Fuentes Dinámicas Externas (`notebooks/10_EDA_trafico_y_meteo.ipynb`):**
  - Descargar y auditar datos abiertos de **estado del tráfico en tiempo real del Ayuntamiento de Madrid** (intensidad, ocupación, carga y nivel de servicio por sensor `lat/lon`) y **datos meteorológicos** (precipitación, viento, temperatura).
  - Implementar cruce espacial por proximidad (sensor de tráfico más cercano a cada parada EMT) y cruce temporal por ventana horaria.
- [ ] **Ingesta en Tiempo Real EMT (`API MobilityLabs`):**
  - Capturar llegadas reales en parada, calcular la variable objetivo (`retraso_real_seg`) y ensamblar el *dataset* final de entrenamiento para el modelo predictivo.

## 🚀 Checklist de Publicación (Paso de Privado a Público en GitHub)
- [ ] **Cambiar visibilidad en GitHub:** Ir a `Settings -> General -> Danger Zone -> Change repository visibility -> Make public`.
- [ ] **Activar GitHub Pages:** Ir a `Settings -> Pages -> Build and deployment -> Deploy from a branch` y seleccionar la rama `main` (`/root`).
- [ ] **Actualizar enlaces interactivos en los Notebooks (`GitHub Pages`):**
  - En `notebooks/02_EDA_stops.ipynb`: Verificar que la URL `https://github.io` carga correctamente el mapa de paradas y eliminar el aviso *(PENDIENTE DE ACTIVAR GITHUB PAGES AL HACER PÚBLICO EL REPOSITORIO)*.
  - En `notebooks/03_EDA_shapes.ipynb`: Verificar que la URL `https://github.io` carga correctamente el visor de trazados y eliminar el aviso *(PENDIENTE DE ACTIVAR GITHUB PAGES AL HACER PÚBLICO EL REPOSITORIO)*.
- [x] **Auditoría visual final en GitHub:** Confirmar que los 8 notebooks (`01` a `08`) renderizan todos sus gráficos estáticos sin espacios en blanco.
