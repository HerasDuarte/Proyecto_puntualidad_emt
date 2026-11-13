import sys
import logging
import pandas as pd
from pathlib import Path

# Calculamos la raíz del proyecto y la añadimos a los directorios de búsqueda de Python
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.limpieza import (
    limpiar_routes, 
    limpiar_stops, 
    limpiar_shapes, 
    limpiar_calendar, 
    limpiar_calendar_dates,
    limpiar_trips,
    limpiar_frequencies,
    limpiar_stop_times  # Motor Big Data en Polars integrado de forma síncrona
)

def ejecutar_pipeline():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Iniciando el pipeline de limpieza y control de calidad GTFS...")

    # CÁLCULO DINÁMICO DE RUTAS ABSOLUTAS
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = BASE_DIR / "data" / "raw" / "gtfs"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    # Aseguramos que la carpeta de salida exista en el entorno local
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # TIER 1 & 2: PROCESAMIENTO DE MAESTROS PREVIOS
    # ==========================================
    
    # 1. Limpieza de Rutas
    logging.info("Procesando 'routes.txt'...")
    routes_raw = pd.read_csv(RAW_DIR / "routes.txt", encoding='utf-8')
    routes_clean = limpiar_routes(routes_raw, output_dir=PROCESSED_DIR)
    routes_clean.to_csv(PROCESSED_DIR / "routes_limpio.csv", index=False, encoding='utf-8')

    # 2. Limpieza de Paradas
    logging.info("Procesando 'stops.txt'...")
    stops_raw = pd.read_csv(RAW_DIR / "stops.txt", encoding='utf-8')
    stops_clean = limpiar_stops(stops_raw, output_dir=PROCESSED_DIR)
    stops_clean.to_csv(PROCESSED_DIR / "stops_limpio.csv", index=False, encoding='utf-8')

    # 3. Limpieza de Trazados (Shapes)
    logging.info("Procesando 'shapes.txt'...")
    shapes_raw = pd.read_csv(RAW_DIR / "shapes.txt", encoding='utf-8')
    shapes_clean = limpiar_shapes(shapes_raw, output_dir=PROCESSED_DIR)
    shapes_clean.to_csv(PROCESSED_DIR / "shapes_limpio.csv", index=False, encoding='utf-8')

    # 4. Limpieza de Calendario Regular
    logging.info("Procesando 'calendar.txt'...")
    calendar_raw = pd.read_csv(RAW_DIR / "calendar.txt", encoding='utf-8')
    calendar_clean = limpiar_calendar(calendar_raw, output_dir=PROCESSED_DIR)
    calendar_clean.to_csv(PROCESSED_DIR / "calendar_limpio.csv", index=False, encoding='utf-8')

    # 5. Limpieza de Excepciones de Calendario
    logging.info("Procesando 'calendar_dates.txt'...")
    calendar_dates_raw = pd.read_csv(RAW_DIR / "calendar_dates.txt", encoding='utf-8')
    calendar_dates_clean = limpiar_calendar_dates(calendar_dates_raw, output_dir=PROCESSED_DIR)
    calendar_dates_clean.to_csv(PROCESSED_DIR / "calendar_dates_limpio.csv", index=False, encoding='utf-8')

    # ==========================================
    # TIER 3: CONECTOR CENTRAL OPERATIVO
    # ==========================================
    
    # 6. Limpieza e Integridad Referencial de Viajes
    logging.info("Procesando 'trips.txt'...")
    trips_raw = pd.read_csv(RAW_DIR / "trips.txt", encoding='utf-8')
    trips_clean = limpiar_trips(trips_raw, routes_clean, calendar_clean, shapes_clean, output_dir=PROCESSED_DIR)
    trips_clean.to_csv(PROCESSED_DIR / "trips_limpio.csv", index=False, encoding='utf-8')

    # 7. Limpieza y Transformación Temporal de Frecuencias (Frequencies)
    logging.info("Procesando 'frequencies.txt'...")
    frequencies_raw = pd.read_csv(RAW_DIR / "frequencies.txt", encoding='utf-8')
    frequencies_clean = limpiar_frequencies(frequencies_raw, trips_clean, output_dir=PROCESSED_DIR)
    frequencies_clean.to_csv(PROCESSED_DIR / "frequencies_limpio.csv", index=False, encoding='utf-8')

    # 8. Limpieza Transaccional Masiva de Horarios de Paso (Stop Times)
    logging.info("Procesando 'stop_times.txt' mediante motor Big Data de Polars...")
    # La función lee directamente de RAW_DIR y aplica el Semi Join contra trips_limpio.csv
    stop_times_clean = limpiar_stop_times(output_dir=PROCESSED_DIR)
    # Persistencia paralela optimizada en Rust a CSV de producción
    stop_times_clean.write_csv(PROCESSED_DIR / "stop_times_limpio.csv")

    logging.info("¡Pipeline ejecutado con éxito! Todos los archivos, incluido el coloso de stop_times, han sido procesados y persistidos.")

if __name__ == "__main__":
    ejecutar_pipeline()
