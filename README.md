# 🚌 Predicción de Puntualidad en la Red de Autobuses de la EMT de Madrid

Sistema de ingeniería de datos y aprendizaje automático (*Machine Learning*) diseñado para modelar, auditar y predecir los retrasos en la red de transporte público de superficie de Madrid (**EMT Madrid**), integrando el catálogo planificado oficial **GTFS** con arquitectura híbrida de alto rendimiento (**Pandas + PyArrow** y **Polars Lazy**).

---

## 🏗️ Arquitectura del Proyecto

El repositorio sigue una estructura modular de ingeniería de software que separa estrictamente el laboratorio exploratorio (`notebooks/`) del motor de producción (`src/`), los scripts ejecutables (`scripts/`), los artefactos visuales (`reports/`), los diccionarios de datos (`docs/`) y las pruebas automatizadas (`tests/`):

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
├── notebooks/                     # Análisis Exploratorio de Datos (EDA) reproducible
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
├── backlog_ideas.md               # Hoja de ideas y mejoras futuras del sistema
├── TODO.md                        # Checklist operativo de tareas previas a publicación
├── requirements.txt               # Dependencias fijadas del entorno virtual
└── README.md
```

---

## 📊 Inventario de Datos Procesados y Tabla Maestra

El pipeline de ingeniería procesa y valida **8 tablas relacionales** y las consolida en una única **Tabla Analítica Maestra** comprimida en formato columnar (`Parquet ZSTD`). *(Nota: Los volúmenes de filas mostrados corresponden a la instantánea base auditada; el pipeline y los tests validan la cardinalidad de forma dinámica ante cualquier actualización del catálogo GTFS).*

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
| **`maestro_gtfs.parquet`** | **Polars + ZSTD** | **2.232.015** | **37** | **Golden Dataset unificado (27,12 MB en disco)** listo para cruce con datos en tiempo real y modelado ML. |

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

- [x] **Fase 1 — Auditoría Exploratoria (EDA):** Análisis forense, visualización corporativa, cartografía estática/interactiva (`reports/`) y diccionarios de datos de las 8 tablas GTFS (`notebooks/01` a `08`).
- [x] **Fase 2 — Ingeniería de Datos y Producción:** Pipeline híbrido (`PyArrow` + `Polars Lazy`), modularización DRY (`src/utils.py`), generación de `maestro_gtfs.parquet` y validación dinámica con `pytest`.
- [ ] **Fase 3 — Ingesta en Tiempo Real (API MobilityLabs EMT):** Captura de tiempos reales de llegada en parada y cálculo de la variable objetivo (`retraso_real_seg`).
- [ ] **Fase 4 — Modelado Predictivo y Evaluación:** Entrenamiento y optimización de modelos supervisados (`LightGBM` / `XGBoost`) para predicción de retrasos por tramo y franja horaria.
