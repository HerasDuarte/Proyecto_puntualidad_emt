# 🚌 Predicción de Puntualidad en la Red de Autobuses de la EMT de Madrid
### Estudio Comparativo mediante Enriquecimiento Progresivo de Datos (*GTFS + Tráfico Urbano + Meteorología*)

Sistema de ingeniería de datos y aprendizaje automático (*Machine Learning*) diseñado para auditar, modelar y predecir los retrasos en la red de transporte público de superficie de Madrid (**EMT Madrid**).

El proyecto sigue una **metodología experimental incremental en tres etapas**, entrenando y comparando **tres modelos predictivos progresivos** sobre un mismo conjunto de validación para medir con rigor el impacto real que aporta cada nueva fuente de información abierta:

1. **Modelo 1 — Baseline Estructural (`GTFS`):** Predicción basada exclusivamente en la oferta planificada, topología vial, saturación simultánea por marquesina en ventanas de 15 minutos, sinuosidad normalizada y contención espacial dentro de la **M-30**.
2. **Modelo 2 — Contexto Vial (`GTFS + Tráfico Ayuntamiento de Madrid`):** Incorporación del estado del tráfico urbano en tiempo real (intensidad, ocupación, carga vial y nivel de servicio ponderados mediante *Inverse Distance Weighting* alrededor de cada parada).
3. **Modelo 3 — Multimodal Completo (`GTFS + Tráfico + Meteorología`):** Integración de variables climáticas cada 15 minutos (precipitación, nieve, visibilidad, rachas de viento y temperatura) para capturar su efecto multiplicador sobre la velocidad comercial y los tiempos de embarque.

---

## 🏗️ Arquitectura del Proyecto

El repositorio sigue una estructura modular de ingeniería de software que separa el laboratorio exploratorio (`notebooks/`) del motor de producción (`src/`), los scripts ejecutables (`scripts/`), los artefactos cartográficos (`reports/`), los diccionarios de datos (`docs/`) y las pruebas automatizadas (`tests/`):

    Proyecto_puntualidad_emt/
    ├── .github/
    │   └── workflows/
    │       ├── ci_tests.yml                 # Integración continua automática (ruff + pytest)
    │       └── sincronizar_nube_hf.yml      # Archivado diario Cloudflare D1 -> Hugging Face
    ├── cloudflare_worker/
    │   ├── src/index.ts                     # Orquestador serverless 24/7 (EMT + Tráfico + AEMET)
    │   ├── schema.sql                       # Esquema relacional de Cloudflare D1
    │   └── wrangler.toml                    # Configuración de Cron Triggers y bindings D1
    ├── data/
    │   ├── raw/
    │   │   ├── gtfs/                        # 8 tablas originales GTFS vigentes de EMT Madrid
    │   │   ├── gtfs_eda_snapshot/           # Instantánea congelada para reproducibilidad exacta de los EDA
    │   │   ├── versiones_gtfs/              # Snapshots históricos crudos de versiones GTFS
    │   │   ├── sensores_trafico/            # Instantánea estática de auditoría (sensores.csv)
    │   │   └── realtime/                    # Capturas diarias en Parquet (emt/, trafico/, meteorologia/)
    │   └── processed/
    │       ├── *_limpio.csv                 # 8 tablas GTFS saneadas
    │       ├── m30_oficial.geojson          # Polígono oficial de la circunvalación M-30
    │       ├── maestro_gtfs.parquet         # Tabla Analítica Maestra consolidada (46 columnas)
    │       ├── cuarentena/                  # Registros huérfanos aislados pendientes de reconciliación
    │       └── versiones_gtfs/              # Historial JSON y copias versionadas de maestro_gtfs
    ├── docs/                                # Diccionarios de datos auditados (01 a 10)
    ├── notebooks/                           # EDAs estructurales y espaciales (01 a 10)
    ├── reports/                             # Mapas estáticos PNG e interactivos HTML
    ├── src/
    │   ├── config.py                        # Configuración centralizada de rutas, URLs y constantes
    │   ├── utils.py                         # Utilidades matemáticas (Haversine), Regex y Parquet DRY
    │   ├── limpieza.py                      # Saneamiento GTFS, cuarentena y Feature Engineering
    │   ├── consolidacion.py                 # Ensamblaje relacional en Polars Lazy -> Parquet ZSTD
    │   ├── sensores_trafico.py              # Ingesta XML tráfico Madrid y cruce espacial IDW
    │   ├── meteorologia.py                  # Ingesta meteorológica cada 15 min (AEMET / Open-Meteo)
    │   └── emt_realtime.py                  # Recolector rotativo EMT (4.928 paradas) con control de cuota
    ├── scripts/
    │   ├── pipeline_limpieza.py             # Ejecuta la limpieza completa de las 8 tablas GTFS
    │   ├── construir_maestro_gtfs.py        # Genera la Tabla Analítica Maestra en Parquet
    │   ├── capturar_trafico_madrid.py       # Descarga instantánea de sensores del Ayuntamiento
    │   ├── test_conexion_api.py             # Verifica credenciales y consulta en vivo MobilityLabs
    │   ├── ejecutar_ingesta_realtime.py     # Orquestador continuo multi-fuente en local
    │   ├── sincronizar_desde_cloudflare.py  # Sincronización D1 -> local, auditoría GTFS y cuarentenas
    │   ├── archivar_nube_a_hf.py            # Volcado transaccional D1 -> Hugging Face + snapshots GTFS
    │   └── descargar_historico_hf.py        # Descarga incremental desde Hugging Face para modelado
    ├── tests/
    │   └── test_pipeline.py                 # Suite de 21 pruebas unitarias y de integración (pytest)
    ├── pyproject.toml                       # Empaquetado PEP 621 y configuración de pytest / ruff
    ├── requirements.txt                     # Dependencias de producción multiplataforma
    ├── requirements-dev.txt                 # Dependencias de testing, cartografía, notebooks y ML
    ├── .env.example                         # Plantilla pública de variables de entorno
    ├── TODO.md                              # Checklist operativo de desarrollo técnico
    └── README.md

```mermaid
flowchart TB
    subgraph F["1. Fuentes de Datos (Madrid)"]
        F1["EMT Madrid OpenData<br/>Llegadas en tiempo real y GTFS"]
        F2["Ayuntamiento de Madrid<br/>Sensores de tráfico (cada 5 min)"]
        F3["AEMET OpenData<br/>Observación meteorológica horaria"]
    end

    subgraph E["2. Ingesta Autónoma 24/7 (Cloudflare Edge)"]
        W["Cloudflare Workers<br/>Cron Triggers sin servidor"]
        D1[("Cloudflare D1<br/>SQLite transaccional en la nube")]
        W --> D1
    end

    subgraph P["3. Orquestación, Calidad y Lakehouse"]
        GA["GitHub Actions<br/>CI/CD (Ruff + 21 Pytest) y ETL"]
        ETL["Motor Analítico Python<br/>Polars Lazy · PyArrow · GeoPandas (IDW)"]
        HF[("Hugging Face Hub<br/>Datasets Parquet + Auditoría SHA-256")]
        GA --> ETL --> HF
    end

    subgraph C["4. Explotación y Visualización"]
        EDA["Notebooks EDA Reproducibles<br/>Instantánea congelada (gtfs_eda_snapshot)"]
        ML["Modelado ML y Dashboard en Vivo<br/>Predicción de puntualidad EMT"]
    end

    F1 & F2 & F3 --> W
    D1 --> GA
    HF --> EDA & ML

    style F fill:#eef6ff,stroke:#2b6cb0,color:#1a202c
    style E fill:#fff7ed,stroke:#c05621,color:#1a202c
    style P fill:#f0fff4,stroke:#2f855a,color:#1a202c
    style C fill:#faf5ff,stroke:#6b46c1,color:#1a202c
```

---

## 📊 Inventario de Datos Procesados (Capa Base GTFS)

El pipeline de ingeniería procesa y valida **8 tablas relacionales** y las consolida en una única **Tabla Analítica Maestra** comprimida en formato columnar (`Parquet ZSTD`), que constituye la estructura base sobre la que se entrenará el **Modelo 1** y se acoplarán las fuentes dinámicas de los **Modelos 2 y 3**.

| Tabla / Archivo Procesado | Motor | Filas Base | Columnas | Características Clave y Predictores Inyectados |
| :--- | :--- | :---: | :---: | :--- |
| **`routes_limpio.csv`** | Pandas + PyArrow | 236 | 6 | Rescate de líneas circulares (`C1`, `C2`), `cabecera_ruta`, `destino_ruta`, `regimen_horario`. |
| **`stops_limpio.csv`** | Pandas + PyArrow | 4.928 | 8 | Coordenadas `WGS84`, `calle_principal` (Regex centralizado), contención espacial M-30 (`en_alta_densidad_real`). |
| **`shapes_limpio.csv`** | Pandas + PyArrow | 193.643 | 8 | Trazados geométricos, `delta_dist_metros`, `indice_sinuosidad` (Haversine), `puntos_por_km`. |
| **`calendar_limpio.csv`** | Pandas + PyArrow | 5 | 10 | Patrones operativos semanales (`LA`, `SA`, `FE`) y ventanas de vigencia. |
| **`calendar_dates_limpio.csv`** | Pandas + PyArrow | 19 | 3 | Excepciones de servicio, refuerzos especiales y días festivos. |
| **`trips_limpio.csv`** | Pandas + PyArrow | 86.871 | 8 | Expediciones activas, rotación de turnos físicos (`block_id`) y `volumen_viajes_bloque`. |
| **`stop_times_limpio.csv`** | **Polars Lazy** | 2.232.015 | 11 | Malla horaria masiva, `tiempo_tramo_teorico_seg`, `distancia_tramo_m`, `progreso_viaje_pct`, `densidad_paradas_acumuladas`. |
| **`frequencies_limpio.csv`** | Pandas + PyArrow | 86.871 | 9 | Cadencias dinámicas, `autobuses_por_hora`, `duracion_franja_secs`, `hora_central_franja`. |
| **`maestro_gtfs.parquet`** | **Polars + ZSTD** | **2.232.015** | **46** | **Golden Dataset estructural (69,30 MB)** enriquecido con `tipo_jornada`, `velocidad_teorica_kmh`, `saturacion_marquesina_15min`, `friccion_acumulada_m30`, `indice_sinuosidad_normalizado` y codificación cíclica `hora_seno`/`hora_coseno`. |

---

## 🧪 Estrategia Experimental: Comparativa de 3 Modelos

| Iteración | Conjunto de Datos | Fuentes Integradas | Pregunta de Negocio que Responde |
| :--- | :--- | :--- | :--- |
| **Modelo 1** | *Baseline Estructural* | `maestro_gtfs.parquet` + Llegadas reales EMT (target) | ¿Cuánto retraso se explica únicamente por la estructura de la red, la hora y la saturación planificada? |
| **Modelo 2** | *Contexto Vial* | Modelo 1 + **Tráfico Ayuntamiento de Madrid (`src/sensores_trafico.py`)** | ¿En qué porcentaje se reduce el error (`MAE` / `RMSE`) al incorporar la intensidad y carga real del tráfico cercano? |
| **Modelo 3** | *Multimodal Completo* | Modelo 2 + **Meteorología 15 min (`src/meteorologia.py`)** | ¿Cuál es la ganancia marginal adicional al añadir lluvia, visibilidad, viento y temperatura al modelo? |

---

## 🚀 Instalación y Ejecución del Pipeline

### 1. Preparación del Entorno Virtual y Credenciales

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1   # En Windows PowerShell
    pip install -r requirements-dev.txt

*(Crea un entorno virtual aislado en Python, lo activa e instala todas las dependencias del proyecto junto con las herramientas de laboratorio y testing. Después, copia `.env.example` a `.env` e introduce tus credenciales de MobilityLabs EMT, Cloudflare y Hugging Face).*

### 2. Ejecución del Pipeline de Limpieza y Tabla Maestra GTFS

    python -m scripts.pipeline_limpieza
    python -m scripts.construir_maestro_gtfs

*(Valida y sanea las 8 tablas relacionales del catálogo GTFS oficial y las ensambla mediante Polars Lazy en el Golden Dataset estructural `data/processed/maestro_gtfs.parquet` de 46 columnas).*

### 3. Ejecución de la Ingesta en Tiempo Real y Sincronización Cloud

    python -m scripts.ejecutar_ingesta_realtime
    python -m scripts.sincronizar_desde_cloudflare

*(Permite capturar en vivo llegadas EMT, tráfico y meteorología desde local o descargar los lotes acumulados por el orquestador serverless 24/7 de Cloudflare D1, auditando cambios diarios del catálogo GTFS y reconciliando cuarentenas).*

### 4. Descarga del Histórico Consolidado desde Hugging Face

    python -m scripts.descargar_historico_hf

*(Recupera de forma incremental todos los ficheros `.parquet` diarios consolidados en la nube y las instantáneas oficiales `gtfs_YYYYMMDD.zip` hacia `data/raw/` para reproducir el entrenamiento de los modelos).*

### 5. Ejecución de Tests Automatizados

    pytest

*(Lanza la suite completa de 21 pruebas unitarias y de integración definida en `tests/test_pipeline.py` aplicando la configuración estándar de `pyproject.toml`).*


---

## 📡 Arquitectura de Captura Continua 24/7 (Cloudflare Workers + D1)

* **Inicio oficial de la campaña de recolección:** **`18/11/2026 a las 18:30 CET`**, operando de forma ininterrumpida **24/7 en la nube** desde el **`19/11/2026`** mediante **Cloudflare Workers (`TypeScript`) + Cloudflare D1 (`SQLite`)**.
* **Cadencia sincronizada multi-fuente:**
  * **EMT Madrid (`*/3 * * * *`):** Barrido rotativo estratificado continuo las **24 horas** (`10` paradas/lote = `4.800` paradas/día, recorriendo la red completa de **4.928 paradas** cada ~24,6h con reinicio automático diario UTC).
  * **Tráfico Municipal (`*/5 * * * *`):** Captura del boletín en vivo del Ayuntamiento de Madrid comprimido en **GZIP** y saneado al sincronizar.
  * **Meteorología Madrid (`*/15 * * * *`):** Observaciones instrumentales reales de la estación **AEMET OpenData (`3195` Madrid-Retiro)** con respaldo automático (`7Timer!`).

* **Sincronización transaccional (`scripts/sincronizar_desde_cloudflare.py`):** Descarga los registros acumulados desde D1, los anexa deduplicados en `data/raw/realtime/*.parquet` y purga únicamente los IDs ya verificados en disco.

* **Estado actual — Fase de Recolección Longitudinal (`Noviembre 2026 – Enero 2027`):** El proyecto se encuentra en campaña activa de adquisición de datos reales 24/7 —con sincronización diaria automática nube-a-local y auditoría criptográfica SHA-256 diaria sobre las 8 tablas operativas GTFS (`routes`, `stops`, `calendar`, `calendar_dates`, `trips`, `frequencies`, `shapes`, `stop_times`) para reconstruir versiones ante cambios de calendario navideño o anual— antes de ejecutar los EDAs dinámicos (`11` y `12`) y el entrenamiento comparativo de los modelos (`1`, `2` y `3`) sobre una muestra libre de sesgo estacional.

* **Archivado Autónomo Cloud-to-Cloud (`GitHub Actions` → `Hugging Face`):** Un flujo programado diario (`02:30 UTC`) vuelca y deduplica los lotes acumulados en **Cloudflare D1** hacia el Dataset privado **`HerasDuarte/emt-madrid-realtime-data`** en formato `.parquet` comprimido y purga la base transaccional, permitiendo capturar de forma 100% autónoma durante toda la ventana longitudinal (**Noviembre 2026 – Enero 2027**) sin depender de ningún ordenador encendido.


---

## 🗺️ Hoja de Ruta (Roadmap)

- [x] **Fase 1 — Auditoría Exploratoria (EDA GTFS):** Análisis forense, visualización corporativa, cartografía estática/interactiva (`reports/`) y diccionarios de datos de las 8 tablas GTFS (`notebooks/01` a `08`).
- [x] **Fase 2 — Ingeniería de Datos Estructural y Tabla Maestra:** Pipeline híbrido (`PyArrow` + `Polars Lazy`), modularización DRY (`src/utils.py`), auditoría relacional (`09_EDA_maestro.ipynb`), generación de `maestro_gtfs.parquet` (46 columnas) y validación dinámica con `pytest`.
- [x] **Fase 3A — Auditoría Espacial de Tráfico y Orquestación Serverless 24/7:** Ingesta XML del Ayuntamiento de Madrid (`10_EDA_sensores_trafico.ipynb`), motor espacial *IDW* (`src/sensores_trafico.py`), despliegue continuo en **Cloudflare Workers + D1** (`cloudflare_worker/`) y sincronizador seguro a Parquet (`scripts/sincronizar_desde_cloudflare.py`).
- [ ] **Fase 3B — EDAs Dinámicos (`11_EDA_meteorologia.ipynb` y `12_EDA_llegadas_emt.ipynb`) y Modelo 1 (*Baseline GTFS*):** Auditoría de las capturas en vivo, cálculo de la variable objetivo (`retraso_real_seg`) contra `maestro_gtfs.parquet` y entrenamiento del primer modelo base.
- [ ] **Fase 4 — Entrenamiento y Evaluación del Modelo 2 (*Contexto Vial*):** Acoplamiento temporal y espacial de las métricas de `src/sensores_trafico.py` sobre el conjunto experimental y medición de ganancia respecto al Modelo 1.
- [ ] **Fase 5 — Enriquecimiento Meteorológico, Modelo 3 y Evaluación Comparativa:** Integración de variables climáticas de 15 min, entrenamiento del tercer modelo multimodal y auditoría comparativa final (`13_Comparativa_Modelos.ipynb`).

---

## 📜 Atribución de Fuentes de Datos

* **Transporte Público y Tiempos de Llegada (GTFS y API Realtime):** *Powered by [EMT de Madrid](http://emtmadrid.es)* (Portal de Datos Abiertos MobilityLabs).
* **Tráfico Urbano:** Datos Abiertos del [Ayuntamiento de Madrid](https://madrid.es) (`informo.madrid.es`).
* **Meteorología:** Agencia Estatal de Meteorología — [AEMET OpenData](https://aemet.es) (Estación `3195` Madrid-Retiro).

---

## 📄 Licencia

Este proyecto (código fuente, notebooks exploratorios, arquitectura cloud y diccionarios de datos) se distribuye bajo la **Licencia MIT** — Copyright (c) 2026 **Mª Carmen Heras Duarte (`@HerasDuarte`)**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
