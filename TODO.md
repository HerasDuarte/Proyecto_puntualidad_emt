# ✅ Lista de Tareas Pendientes (`TODO`)

Registro operativo de tareas de ingeniería de datos, captura en tiempo real y modelado predictivo.

> **📡 Bitácora de Despliegue Continuo en Tiempo Real (24/7):**
> * **Inicio oficial de ingesta sincronizada (`EMT + Tráfico + Meteorología`):** `18/11/2026 a las 18:30 CET` mediante orquestador local (`scripts/ejecutar_ingesta_realtime.py`).
> * **Migración a Arquitectura Serverless 24/7 (`19/11/2026`):** Desplegado en producción continua sobre **Cloudflare Workers (`TypeScript`) + Cloudflare D1 (`SQLite`)** con `Cron Triggers` exactos (`*/3` EMT con reinicio diario UTC de `4.500` reqs/día, `*/5` Tráfico municipal comprimido con GZIP y `*/15` Meteorología) y sincronización transaccional a `.parquet` locales mediante `scripts/sincronizar_desde_cloudflare.py`.
> * **Mejora climática pendiente:** Incorporar estación instrumental real **AEMET OpenData (`3195` Madrid-Retiro)** como fuente primaria en `cloudflare_worker/src/index.ts` en cuanto se active la clave de correo institucional, manteniendo `7Timer!` como respaldo automático.

## 📊 Hoja de Trabajo Técnica (Fases 3 a 5)
- [x] **Cierre de Capa Estructural (`notebooks/09_EDA_maestro.ipynb`):**
  - Realizado análisis exploratorio sobre `maestro_gtfs.parquet` (2.232.015 tramos unificados) e implementado *Feature Engineering* relacional estructural (`tipo_jornada`, `indice_sinuosidad_normalizado`, `velocidad_teorica_kmh`, `saturacion_marquesina_15min`, `friccion_acumulada_m30`, `hora_seno` y `hora_coseno`) en `src/consolidacion.py`.
- [x] **Auditoría Espacial de Tráfico Urbano (`notebooks/10_EDA_trafico.ipynb`):**
  - Auditado el boletín en vivo de **4.904 puntos de medida** del Ayuntamiento de Madrid (`4.362` operativos), generando `docs/10_trafico_Dict.md` y `reports/mapa_sensores_trafico.html`, e implementado el cruce geodésico *IDW* en `src/trafico.py`.
- [x] **Arquitectura de Ingesta Sincronizada Multi-Fuente 24/7 (`Local + Serverless Cloud`):**
  - Implementados conectores locales (`src/meteorologia.py`, `src/emt_realtime.py`, `scripts/ejecutar_ingesta_realtime.py`) validados en `tests/test_pipeline.py` (`9 passed`).
  - Desplegado el orquestador autónomo 24/7 en **Cloudflare Workers + D1** (`cloudflare_worker/src/index.ts`) y el sincronizador seguro nube-a-disco (`scripts/sincronizar_desde_cloudflare.py`).
- [ ] **EDAs de Datos Dinámicos y Modelo 1 (*Baseline Estructural GTFS*):**
  - Construir `notebooks/11_EDA_meteorologia.ipynb` (`docs/11_meteorologia_Dict.md`) y `notebooks/12_EDA_llegadas_emt.ipynb` (`docs/12_emt_realtime_Dict.md`).
  - Calcular la variable objetivo (`retraso_real_seg`) cruzando llegadas reales con `maestro_gtfs.parquet` y entrenar el **Modelo 1**, fijando el conjunto de validación (*test set*) y las métricas base (`MAE`, `RMSE`, `R²`).
- [ ] **Modelo 2 — Entrenamiento y Evaluación con Contexto Vial (`GTFS + Tráfico`):**
  - Entrenar el **Modelo 2** incorporando las 7 variables espaciales de `src/trafico.py` mediante cruce temporal (`merge_asof`) sobre el mismo conjunto de validación y cuantificar la reducción de error respecto al Modelo 1.
- [ ] **Modelo 3 — Enriquecimiento Meteorológico y Comparativa Final (`13_Comparativa_Modelos.ipynb`):**
  - Entrenar el **Modelo 3** acoplando las variables meteorológicas de 15 min y construir el estudio comparativo final enfrentando los 3 modelos e interpretando la importancia de variables (*SHAP / Feature Importance*).
