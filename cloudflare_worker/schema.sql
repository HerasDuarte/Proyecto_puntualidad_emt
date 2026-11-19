-- Esquema de almacenamiento crudo en Cloudflare D1 (emt-madrid-db)

CREATE TABLE IF NOT EXISTS emt_llegadas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp_captura_utc TEXT NOT NULL,
    stop_id TEXT NOT NULL,
    line TEXT,
    destination TEXT,
    bus_id TEXT,
    estimate_arrive_sec REAL,
    distance_bus_m REAL,
    position_type_bus TEXT,
    is_head TEXT
);

CREATE TABLE IF NOT EXISTS trafico_xml_raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_captura_utc TEXT NOT NULL,
    xml_gzip_base64 TEXT NOT NULL,
    tamano_bytes_original INTEGER
);

CREATE TABLE IF NOT EXISTS meteo_madrid (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_captura_utc TEXT NOT NULL,
    timestamp_api_madrid TEXT,
    temperatura_c REAL,
    sensacion_termica_c REAL,
    humedad_relativa_pct REAL,
    es_de_dia INTEGER,
    precipitacion_mm REAL,
    lluvia_mm REAL,
    nevadas_cm REAL,
    visibilidad_m REAL,
    codigo_tiempo_wmo INTEGER,
    viento_velocidad_kmh REAL,
    viento_rachas_kmh REAL
);

CREATE TABLE IF NOT EXISTS estado_cuota (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    fecha_cuota_utc TEXT NOT NULL,
    session_id TEXT NOT NULL,
    peticiones_realizadas INTEGER NOT NULL DEFAULT 0,
    indice_parada_actual INTEGER NOT NULL DEFAULT 0,
    en_pausa_cuota INTEGER NOT NULL DEFAULT 0,
    access_token TEXT,
    token_expira_utc TEXT,
    ultima_actualizacion_utc TEXT
);
