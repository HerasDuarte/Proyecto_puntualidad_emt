# 🚌 Predicción de Puntualidad en la Red de Autobuses de la EMT de Madrid
### Estudio Comparativo mediante Enriquecimiento Progresivo de Datos (*GTFS + Tráfico Urbano + Meteorología*)

Sistema de ingeniería de datos y aprendizaje automático (*Machine Learning*) diseñado para auditar, modelar y predecir los retrasos en la red de transporte público de superficie de Madrid (**EMT Madrid**).

El proyecto sigue una **metodología experimental incremental en tres etapas**, entrenando y comparando **tres modelos predictivos progresivos** sobre un mismo conjunto de validación para medir con rigor el impacto real que aporta cada nueva fuente de información abierta:

1. **Modelo 1 — Baseline Estructural (`GTFS`):** Predicción basada exclusivamente en la oferta planificada, topología vial, densidad de marquesinas, sinuosidad de trazado y contención espacial dentro de la **M-30**.

2. **Modelo 2 — Contexto Vial (`GTFS + Tráfico Ayuntamiento de Madrid`):** Incorporación del estado del tráfico urbano en tiempo real (intensidad, ocupación, carga vial y nivel de servicio en espiras electromagnéticas próximas a cada parada).

3. **Modelo 3 — Multimodal Completo (`GTFS + Tráfico + Meteorología`):** Integración de variables climáticas horarias (precipitación, rachas de viento y temperatura) para capturar su efecto multiplicador sobre la velocidad comercial y los tiempos de embarque.

---

## 🏗️ Arquitectura del Proyecto

El repositorio sigue una estructura modular de ingeniería de software que separa el laboratorio exploratorio (`notebooks/`) del motor de producción (`src/`), los scripts ejecutables (`scripts/`), los artefactos cartográficos (`reports/`), los diccionarios de datos (`docs/`) y las pruebas automatizadas (`tests/`):

```text
Proyecto_puntualidad_emt/
├── data/
│   ├── raw/gtfs/                  # Ficheros originales GTFS de la EMT de Madrid (sin modificar)
│   └── processed/                 # 8 tablas saneadas (*_limpio.csv) + maestro_gtfs.parquet
├── docs/                          # Diccionarios de datos completos y auditados por tabla
│   ├── 01_routes_Dict.md
│   ├── 02_stops_Dict.md
│   ├── 03_shapes_Dict.md
│   ├── 04_calendar_Dict.md
│   ├── 05_calendar_dates_Dict.md
│   ├── 06_trips_Dict.md
│   ├── 07_stop_times_Dict.md
│   └── 08_frequencies_Dict.md
├── notebooks/                     # Análisis Exploratorio de Datos (EDA) y Experimentación ML
│   ├── 01_EDA_routes.ipynb
│   ├── 02_EDA_stops.ipynb
│   ├── 03_EDA_shapes.ipynb
│   ├── 04_EDA_calendar.ipynb
│   ├── 05_EDA_calendar_dates.ipynb
│   ├── 06_EDA_trips.ipynb
│   ├── 07_EDA_stop_times.ipynb
│   └── 08_EDA_frequencies.ipynb
├── reports/                       # Cartografía exportada en PNG estático y HTML interactivo
│   ├── mapa_paradas_emt.html
│   ├── mapa_paradas_emt.png
│   ├── visor_trazado_emt.html
│   └── visor_trazado_emt.png
├── src/                           # Módulos reutilizables de producción
│   ├── utils.py                   # Utilidades matemáticas, geodésicas (Haversine) y Regex
│   ├── limpieza.py                # Motores de saneamiento, cuarentena y Feature Engineering
│   └── consolidacion.py           # Ensamblaje relacional en Polars Lazy -> Parquet ZSTD
├── scripts/                       # Scripts ejecutables de línea de comandos
│   ├── pipeline_limpieza.py       # Ejecuta la limpieza completa de las 8 tablas GTFS
│   └── construir_maestro_gtfs.py  # Genera la Tabla Analítica Maestra en formato Parquet
├── tests/                         # Suite de pruebas unitarias y de integración (pytest)
│   └── test_pipeline.py
├── TODO.md                        # Checklist operativo de desarrollo y publicación
├── requirements.txt               # Dependencias fijadas del entorno virtual
└── README.md
```

---

## 📊 Inventario de Datos Procesados (Capa Base GTFS)

El pipeline de ingeniería procesa y valida **8 tablas relacionales** y las consolida en una única **Tabla Analítica Maestra** comprimida en formato columnar (`Parquet ZSTD`), que constituye la estructura base sobre la que se entrenará el **Modelo 1** y se acoplarán las fuentes dinámicas de los **Modelos 2 y 3**. *(Nota: Los volúmenes de filas mostrados corresponden a la instantánea base auditada; el pipeline y los tests validan la cardinalidad de forma dinámica ante cualquier actualización del catálogo GTFS).*

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
| **`maestro_gtfs.parquet`** | **Polars + ZSTD** | **2.232.015** | **37** | **Golden Dataset estructural (27,12 MB en disco)** listo para cruce con tiempos reales, tráfico y clima. |

---

## 🧪 Estrategia Experimental: Comparativa de 3 Modelos

| Iteración | Conjunto de Datos | Fuentes Integradas | Pregunta de Negocio que Responde |
| :--- | :--- | :--- | :--- |
| **Modelo 1** | *Baseline Estructural* | `maestro_gtfs.parquet` + Llegadas reales EMT (target) | ¿Cuánto retraso se explica únicamente por la estructura de la red, la hora y la saturación planificada? |
| **Modelo 2** | *Contexto Vial* | Modelo 1 + **Tráfico Ayuntamiento de Madrid** | ¿En qué porcentaje se reduce el error (`MAE` / `RMSE`) al incorporar la intensidad y carga real del tráfico cercano? |
| **Modelo 3** | *Multimodal Completo* | Modelo 2 + **Meteorología (AEMET / Open-Meteo)** | ¿Cuál es la ganancia marginal adicional al añadir lluvia, viento y temperatura al modelo? |

---

## 🚀 Instalación y Ejecución del Pipeline

### 1. Preparación del Entorno Virtual
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # En Windows PowerShell
pip install -r requirements.txt
```

### 2. Ejecución del Pipeline de Limpieza y Feature Engineering
Procesa los 8 archivos brutos de `data/raw/gtfs/`, aplica reglas de integridad referencial, aísla cualquier anomalía en archivos `quarantine_*.csv` y exporta los conjuntos limpios en **menos de 3 segundos**:
```bash
python -m scripts.pipeline_limpieza
```

### 3. Construcción de la Tabla Analítica Maestra (`maestro_gtfs.parquet`)
Ejecuta el ensamblaje relacional en **Polars Lazy** verificando dinámicamente la preservación íntegra del 100% de los registros de `stop_times_limpio.csv` sin duplicidades ni nulos en claves críticas:
```bash
python -m scripts.construir_maestro_gtfs
```

### 4. Ejecución de Tests Automatizados
Valida las reglas de negocio, el aislamiento en cuarentena y la integridad dinámica del fichero `.parquet` generado:
```bash
pytest -v
```

---

## 🗺️ Hoja de Ruta (Roadmap)

- [x] **Fase 1 — Auditoría Exploratoria (EDA GTFS):** Análisis forense, visualización corporativa, cartografía estática/interactiva (`reports/`) y diccionarios de datos de las 8 tablas GTFS (`notebooks/01` a `08`).
- [x] **Fase 2 — Ingeniería de Datos Estructural:** Pipeline híbrido (`PyArrow` + `Polars Lazy`), modularización DRY (`src/utils.py`), generación de `maestro_gtfs.parquet` y validación dinámica con `pytest`.
- [ ] **Fase 3 — EDA de Tabla Maestra y Modelo 1 (*Baseline GTFS*):** Ingeniería de características relacional (`09_EDA_maestro.ipynb`), ingesta de tiempos reales EMT (`API MobilityLabs`) y entrenamiento del primer modelo base.
- [ ] **Fase 4 — Enriquecimiento con Tráfico Urbano y Modelo 2:** Auditoría de sensores de tráfico del Ayuntamiento de Madrid (`10_EDA_trafico.ipynb`), cruce espacial con paradas EMT y entrenamiento del segundo modelo.
- [ ] **Fase 5 — Enriquecimiento Meteorológico, Modelo 3 y Evaluación Comparativa:** Integración de variables climáticas (`11_EDA_meteorologia.ipynb`), entrenamiento del tercer modelo multimodal y auditoría comparativa final (`12_Comparativa_Modelos.ipynb`).
