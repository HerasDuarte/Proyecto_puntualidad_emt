import pytest
import pandas as pd
import polars as pl
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.limpieza import limpiar_routes, limpiar_frequencies


# ==============================================================================
# 1. PRUEBAS UNITARIAS CON DATOS SINTÉTICOS (LÓGICA DE NEGOCIO Y CUARENTENAS)
# ==============================================================================

def test_limpiar_routes_rescate_circulares_y_regimen():
    """Verifica el rescate de líneas circulares sin destino y la clasificación horaria."""
    df_raw = pd.DataFrame({
        'route_id': ['001', 'C1', 'N01'],
        'agency_id': ['EMT', 'EMT', 'EMT'],
        'route_short_name': ['1', 'C1', 'N1'],
        'route_long_name': ['CRISTO REY - PROSPERIDAD', 'CIRCULAR 1', 'CIBELES - SANCHINARRO'],
        'route_desc': [None, None, None],
        'route_type': list((3, 3, 3)),
        'route_url': ['http://a', 'http://b', 'http://c'],
        'route_color': ['0056B3', '0056B3', '0056B3'],
        'route_text_color': ['FFFFFF', 'FFFFFF', 'FFFFFF']
    })

    df_clean = limpiar_routes(df_raw)

    assert len(df_clean) == 3
    assert df_clean.loc[df_clean['route_short_name'] == 'C1', 'destino_ruta'].item() == 'CIRCULAR 1'
    assert df_clean.loc[df_clean['route_short_name'] == 'N1', 'regimen_horario'].item() == 'Servicio Nocturno (Búho)'
    assert df_clean.loc[df_clean['route_short_name'] == '1', 'regimen_horario'].item() == 'Servicio Diurno Regular'


def test_limpiar_frequencies_cuarentena_y_features(tmp_path):
    """Verifica que frecuencias invertidas o huérfanas van a cuarentena y las válidas se enriquecen."""
    df_trips_validos = pd.DataFrame({'trip_id': ['TRIP_OK']})

    df_freq_raw = pd.DataFrame({
        'trip_id': ['TRIP_OK', 'TRIP_OK', 'TRIP_HUERFANO'],
        'start_time': ['7:00:00', '10:00:00', '8:00:00'],
        'end_time': ['7:59:59', '9:00:00', '8:59:59'],
        'headway_secs': list((300, 600, 900))
    })

    df_clean = limpiar_frequencies(df_freq_raw, df_trips_validos, output_dir=tmp_path)

    assert len(df_clean) == 1
    assert (tmp_path / 'quarantine_frequencies.csv').exists()

    assert df_clean['start_time_secs'].item() == 25200
    assert df_clean['end_time_secs'].item() == 28799
    assert df_clean['duracion_franja_secs'].item() == 3600
    assert float(df_clean['autobuses_por_hora'].item()) == pytest.approx(12.0, rel=1e-3)


# ==============================================================================
# 2. PRUEBAS DE INTEGRACIÓN SOBRE EL PIPELINE PERSISTIDO (DATA/PROCESSED)
# ==============================================================================

def test_existencia_archivos_procesados():
    """Certifica que los 8 CSV limpios y la Tabla Maestra Parquet existen en disco."""
    processed_dir = ROOT_DIR / 'data' / 'processed'
    archivos_esperados = [
        'routes_limpio.csv', 'stops_limpio.csv', 'shapes_limpio.csv',
        'calendar_limpio.csv', 'calendar_dates_limpio.csv', 'trips_limpio.csv',
        'stop_times_limpio.csv', 'frequencies_limpio.csv', 'maestro_gtfs.parquet'
    ]
    for archivo in archivos_esperados:
        assert (processed_dir / archivo).exists(), f"Falta el archivo procesado: {archivo}"


def test_ausencia_cuarentenas_en_produccion():
    """Verifica que el catálogo oficial de la EMT no generó archivos de cuarentena."""
    processed_dir = ROOT_DIR / 'data' / 'processed'
    cuarentenas = list(processed_dir.glob('quarantine_*.csv'))
    assert len(cuarentenas) == 0, f"Se detectaron archivos de cuarentena inesperados: {cuarentenas}"


def test_integridad_tabla_maestra_parquet():
    """Audita dinámicamente que maestro_gtfs.parquet conserve el 100% de stop_times_limpio.csv sin nulos críticos."""
    processed_dir = ROOT_DIR / 'data' / 'processed'
    lf_parquet = pl.scan_parquet(processed_dir / 'maestro_gtfs.parquet')
    lf_stop_times = pl.scan_csv(processed_dir / 'stop_times_limpio.csv', infer_schema=False)

    filas_parquet = lf_parquet.select(pl.len()).collect().item()
    filas_base = lf_stop_times.select(pl.len()).collect().item()
    columnas = lf_parquet.collect_schema().names()

    assert filas_parquet > 0, "La Tabla Maestra Parquet está vacía."
    assert filas_parquet == filas_base, f"Cardinalidad alterada: {filas_parquet:,} vs {filas_base:,}"
    assert len(columnas) == 37, f"Esquema alterado: {len(columnas)} columnas"

    nulos_criticos = (
        lf_parquet.select([
            pl.col('trip_id').null_count(),
            pl.col('stop_id').null_count(),
            pl.col('route_id').null_count(),
            pl.col('stop_lat').null_count(),
            pl.col('stop_lon').null_count()
        ])
        .collect()
        .row(0)
    )
    assert sum(nulos_criticos) == 0, f"Se detectaron valores nulos en claves críticas: {nulos_criticos}"
