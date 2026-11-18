# ✅ Lista de Tareas Pendientes (`TODO`)

Registro operativo de tareas de ingeniería de datos, captura en tiempo real y modelado predictivo.

> **📡 Bitácora de Captura en Tiempo Real:**
> * **Inicio oficial de ingesta sincronizada (`EMT + Tráfico + Meteorología`):** `18/11/2026 a las 18:30 CET` (ejecución inicial en entorno local mediante `scripts/ejecutar_ingesta_realtime.py` con barrido rotativo sobre las 4.928 paradas EMT y control autónomo de cuota de 4.500 reqs/24h).
> * **Arquitectura objetivo en despliegue continuo:** Migración prevista del orquestador y del ciclo periódico de reentrenamiento a un servidor Linux dedicado 24/7 (*Always-Free Cloud VM*) para garantizar captura ininterrumpida en todas las franjas horarias sin sesgo nocturno.

## 📊 Hoja de Trabajo Técnica (Fases 3 a 5)
- [x] **Cierre de Capa Estructural (`notebooks/09_EDA_maestro.ipynb`):**
  - Realizado análisis exploratorio sobre `maestro_gtfs.parquet` (2.232.015 tramos unificados).
  - Implementado *Feature Engineering* relacional estructural (`tipo_jornada`, `indice_sinuosidad_normalizado`, `velocidad_teorica_kmh`, `saturacion_marquesina_15min`, `friccion_acumulada_m30`, `hora_seno` y `hora_coseno`).
  - Llevadas las 9 variables a producción en `src/consolidacion.py`, documentadas en `docs/09_maestro_Dict.md` y validadas con `pytest` (`6 passed`).
- [x] **Auditoría Espacial de Tráfico Urbano (`notebooks/10_EDA_trafico.ipynb`):**
  - Descargado y auditado el boletín en vivo de **4.904 puntos de medida** del Ayuntamiento de Madrid (`4.362` operativos), generando `docs/10_trafico_Dict.md` y el mapa interactivo `reports/mapa_sensores_trafico.html`.
  - Implementado y llevado a producción en `src/trafico.py` el cruce geodésico ponderado (*IDW 500 m* + respaldo *1.000 m*) que cubre el **98,07%** de las paradas EMT (**100%** en el interior de la M-30).
- [x] **Arquitectura Modular de Ingesta Sincronizada Multi-Fuente en Tiempo Real:**
  - Implementado `src/meteorologia.py` (captura cada 15 min de 11 variables climáticas en Madrid vía Open-Meteo).
  - Implementado `src/emt_realtime.py` (barrido rotativo estratificado sobre las 4.928 paradas EMT con tope autónomo de 4.500 peticiones y reactivación programada a las 24h + 1 min).
  - Implementado y verificado en ejecución `scripts/ejecutar_ingesta_realtime.py` con persistencia incremental en Parquet (`data/raw/realtime/`) y tests unitarios en `tests/test_pipeline.py` (`9 passed`).
- [ ] **EDAs de Datos Dinámicos y Modelo 1 (*Baseline Estructural GTFS*):**
  - Construir `notebooks/11_EDA_meteorologia.ipynb` (`docs/11_meteorologia_Dict.md`) y `notebooks/12_EDA_llegadas_emt.ipynb` (`docs/12_emt_realtime_Dict.md`).
  - Calcular la variable objetivo (`retraso_real_seg`) cruzando llegadas reales con `maestro_gtfs.parquet` y entrenar el **Modelo 1**, fijando el conjunto de validación (*test set*) y las métricas base (`MAE`, `RMSE`, `R²`).
- [ ] **Modelo 2 — Entrenamiento y Evaluación con Contexto Vial (`GTFS + Tráfico`):**
  - Entrenar el **Modelo 2** incorporando las 7 variables espaciales de `src/trafico.py` mediante cruce temporal (`merge_asof`) sobre el mismo conjunto de validación y cuantificar la reducción de error respecto al Modelo 1.
- [ ] **Modelo 3 — Enriquecimiento Meteorológico y Comparativa Final (`13_Comparativa_Modelos.ipynb`):**
  - Entrenar el **Modelo 3** acoplando las variables meteorológicas de 15 min (`src/meteorologia.py`) y construir el estudio comparativo final enfrentando los 3 modelos e interpretando la importancia de variables (*SHAP / Feature Importance*).
