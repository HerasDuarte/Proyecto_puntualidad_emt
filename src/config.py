"""
Configuración Centralizada del Proyecto (Rutas, Constantes de Dominio y Esquemas).
Evita la duplicación de literales entre los módulos de src/, scripts/ y tests/.
"""

from pathlib import Path

# Directorios raíz de datos
DIR_DATA = Path("data")
DIR_RAW = DIR_DATA / "raw"
DIR_PROCESSED = DIR_DATA / "processed"

# Subcarpetas estructuradas en data/raw/
DIR_GTFS_ACTIVO = DIR_RAW / "gtfs"
DIR_VERSIONES_GTFS_RAW = DIR_RAW / "versiones_gtfs"
DIR_SENSORES_TRAFICO_RAW = DIR_RAW / "sensores_trafico"
DIR_REALTIME = DIR_RAW / "realtime"
DIR_REALTIME_EMT = DIR_REALTIME / "emt"
DIR_REALTIME_TRAFICO = DIR_REALTIME / "trafico"
DIR_REALTIME_METEO = DIR_REALTIME / "meteorologia"

# Subcarpetas estructuradas en data/processed/
DIR_CUARENTENA = DIR_PROCESSED / "cuarentena"
DIR_VERSIONES_GTFS_PROCESSED = DIR_PROCESSED / "versiones_gtfs"
RUTA_MAESTRO_PARQUET = DIR_PROCESSED / "maestro_gtfs.parquet"
RUTA_HISTORIAL_GTFS_LOCAL = DIR_VERSIONES_GTFS_PROCESSED / "historial_versiones_gtfs.json"

# Catálogo oficial GTFS EMT Madrid (Portal CKAN datos.emtmadrid.es)
URL_GTFS_OFICIAL_EMT = (
    "https://datos.emtmadrid.es/dataset/9b23259a-4491-494b-9695-36a7709b2c12"
    "/resource/3cba2058-9833-422c-a704-bf992d31d2ee/download/gtfs_emt.zip"
)

# Las 8 tablas operativas auditadas en los EDAs 01-08 (excluye agency.txt y feed_info.txt)
TABLAS_GTFS_OPERATIVAS = (
    "routes.txt",
    "stops.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "trips.txt",
    "frequencies.txt",
    "shapes.txt",
    "stop_times.txt",
)

# Repositorio Dataset privado en Hugging Face
HF_DATASET_ID_DEFECTO = "HerasDuarte/emt-madrid-realtime-data"
RUTA_HISTORIAL_GTFS_HF = "gtfs_snapshots/historial_gtfs_nube.json"

# Configuración de particionado y claves de deduplicación para datos en tiempo real
CONFIG_TABLAS_REALTIME = {
    "emt_llegadas": {
        "subcarpeta": "emt",
        "prefijo": "emt_llegadas",
        "dedup": ["timestamp_captura_utc", "stop_id", "line", "bus_id"],
    },
    "trafico_xml_raw": {
        "subcarpeta": "trafico",
        "prefijo": "trafico_sensores",
        "dedup": ["timestamp_captura_utc", "idelem"],
    },
    "meteo_madrid": {
        "subcarpeta": "meteorologia",
        "prefijo": "meteo_madrid",
        "dedup": ["timestamp_captura_utc"],
    },
}
