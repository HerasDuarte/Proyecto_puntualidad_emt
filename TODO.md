# ✅ Lista de Tareas Pendientes (`TODO`)

Registro operativo de tareas de ingeniería de datos, captura en tiempo real y modelado predictivo.

## 📊 Hoja de Trabajo Técnica (Fases 3 a 5)
- [x] **Cierre de Capa Estructural (`notebooks/09_EDA_maestro.ipynb`):**
  - Realizado análisis exploratorio sobre `maestro_gtfs.parquet` (2.232.015 tramos unificados).
  - Implementado *Feature Engineering* relacional estructural (`tipo_jornada`, `indice_sinuosidad_normalizado`, `velocidad_teorica_kmh`, `saturacion_marquesina_15min`, `friccion_acumulada_m30`, `hora_seno` y `hora_coseno`).
  - Llevadas las 9 variables a producción en `src/consolidacion.py`, documentadas en `docs/09_maestro_Dict.md` y validadas con `pytest` (`6 passed`).
- [x] **Auditoría Espacial de Tráfico Urbano y Conectividad API EMT (`notebooks/10_EDA_trafico.ipynb`):**
  - Descargado y auditado el boletín en vivo de **4.904 puntos de medida** del Ayuntamiento de Madrid (`4.362` operativos), generando `docs/10_trafico_Dict.md` y el mapa interactivo `reports/mapa_sensores_trafico.html`.
  - Implementado y llevado a producción en `src/trafico.py` el cruce geodésico ponderado (*IDW 500 m* + respaldo *1.000 m*) que cubre el **98,07%** de las paradas EMT (**100%** en el interior de la M-30).
  - Configuradas credenciales seguras (`.env`) y verificada la conexión en tiempo real con la **API MobilityLabs EMT** (`scripts/test_conexion_api.py`).
- [ ] **Recolector Sincronizado Multi-Fuente y Modelo 1 (*Baseline Estructural GTFS*):**
  - Diseñar e implementar el recolector sincronizado (barrido rotativo estratificado de paradas EMT + instantánea de tráfico + meteorología horaria) para capturar observaciones reales sin agotar límites de cuota.
  - Calcular la variable objetivo (`retraso_real_seg`) cruzando llegadas reales con `maestro_gtfs.parquet` y entrenar el **Modelo 1**, fijando el conjunto de validación (*test set*) y las métricas base (`MAE`, `RMSE`, `R²`).
- [ ] **Modelo 2 — Entrenamiento y Evaluación con Contexto Vial (`GTFS + Tráfico`):**
  - Entrenar el **Modelo 2** incorporando las 7 variables espaciales de `src/trafico.py` sobre el mismo conjunto de validación y cuantificar la reducción de error respecto al Modelo 1.
- [ ] **Modelo 3 — Enriquecimiento Meteorológico y Comparativa Final (`notebooks/11_EDA_meteorologia.ipynb` y `12_Comparativa_Modelos.ipynb`):**
  - Auditar e integrar variables meteorológicas horarias (`Open-Meteo`: precipitación, viento, temperatura, humedad).
  - Entrenar el **Modelo 3** y construir el estudio comparativo final enfrentando los 3 modelos e interpretando la importancia de variables (*SHAP / Feature Importance*).
