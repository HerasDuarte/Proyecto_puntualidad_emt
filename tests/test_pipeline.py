import pytest
import pandas as pd
import polars as pl
from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.limpieza import limpiar_routes, limpiar_frequencies
from src.trafico import sanear_xml_madrid, vincular_trafico_a_paradas
from src.meteorologia import obtener_meteorologia_actual
from src.emt_realtime import ClienteEMTRealtime, construir_lista_paradas_estratificada


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
    assert len(columnas) == 46, f"Esquema alterado: {len(columnas)} columnas (esperadas: 46)"

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


def test_feature_engineering_relacional_maestro():
    """Valida los dominios físicos y ausencia de nulos en las 9 variables de interacción relacional."""
    processed_dir = ROOT_DIR / 'data' / 'processed'
    lf_parquet = pl.scan_parquet(processed_dir / 'maestro_gtfs.parquet')

    nuevas_cols = [
        'hora_entera', 'tramo_15min', 'tipo_jornada',
        'indice_sinuosidad_normalizado', 'velocidad_teorica_kmh',
        'friccion_acumulada_m30', 'hora_seno', 'hora_coseno',
        'saturacion_marquesina_15min'
    ]

    nulos_nuevas = lf_parquet.select([pl.col(c).null_count() for c in nuevas_cols]).collect().row(0)
    assert sum(nulos_nuevas) == 0, f"Se detectaron nulos en las nuevas variables: {nulos_nuevas}"

    metricas = next(iter(lf_parquet.select([
        pl.col('hora_entera').min().alias('min_h'),
        pl.col('hora_entera').max().alias('max_h'),
        pl.col('tramo_15min').min().alias('min_t15'),
        pl.col('tramo_15min').max().alias('max_t15'),
        pl.col('velocidad_teorica_kmh').min().alias('min_vel'),
        pl.col('velocidad_teorica_kmh').max().alias('max_vel'),
        pl.col('friccion_acumulada_m30').min().alias('min_fric'),
        pl.col('friccion_acumulada_m30').max().alias('max_fric'),
        pl.col('saturacion_marquesina_15min').min().alias('min_sat')
    ]).collect().to_dicts()))

    assert 0 <= metricas['min_h'] <= metricas['max_h'] <= 23
    assert 0 <= metricas['min_t15'] <= metricas['max_t15'] <= 95
    assert 0.0 <= metricas['min_vel'] <= metricas['max_vel'] <= 80.0
    assert 0.0 <= metricas['min_fric'] <= metricas['max_fric'] <= 1.0
    assert metricas['min_sat'] >= 1

    jornadas_unicas = set(lf_parquet.select(pl.col('tipo_jornada').unique()).collect()['tipo_jornada'].to_list())
    assert jornadas_unicas.issubset({'LABORABLE', 'VIERNES_VISPERA', 'FIN_DE_SEMANA'})


# ==============================================================================
# 3. PRUEBAS UNITARIAS DE MÓDULOS EN TIEMPO REAL (TRÁFICO, METEO Y CUOTA EMT)
# ==============================================================================

def test_trafico_saneamiento_xml_y_cruce_idw():
    """Verifica el saneamiento XML municipal y la ponderación espacial IDW sobre paradas."""
    xml_sucio = b"<pms><pm><descripcion>VELAZQUEZ & GOYA\x07</descripcion></pm></pms>"
    xml_limpio = sanear_xml_madrid(xml_sucio)
    assert "&amp;" in xml_limpio
    assert "\x07" not in xml_limpio

    df_stops_test = pd.DataFrame({
        "stop_id": ["100", "200"],
        "stop_lat": [40.4200, 40.5000],
        "stop_lon": [-3.6900, -3.6000]
    })

    df_sensores_test = pd.DataFrame({
        "timestamp_trafico": ["18/11/2026 19:00:00", "18/11/2026 19:00:00"],
        "idelem": ["S1", "S2"],
        "carga": [40.0, 80.0],
        "ocupacion": [15.0, 35.0],
        "intensidad": [500.0, 1200.0],
        "nivel_servicio": [1.0, 3.0],
        "lat": [40.4205, 40.4220],
        "lon": [-3.6900, -3.6900]
    })

    df_res = vincular_trafico_a_paradas(df_stops_test, df_sensores_test)

    assert bool(df_res.loc[0, "tiene_cobertura_trafico_500m"]) is True
    assert int(df_res.loc[0, "num_sensores_500m"]) == 2
    assert float(df_res.loc[0, "trafico_nivel_servicio_max"]) == 3.0
    assert 40.0 < float(df_res.loc[0, "trafico_carga_entorno"]) < 60.0

    assert bool(df_res.loc[1, "tiene_cobertura_trafico_500m"]) is False
    assert np.isnan(df_res.loc[1, "trafico_carga_entorno"])


def test_meteorologia_parseo_open_meteo(monkeypatch):
    """Verifica el parseo y esquema de las 13 variables meteorológicas de 15 minutos."""
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "current": {
                    "time": "2026-11-18T19:45",
                    "temperature_2m": 12.4,
                    "apparent_temperature": 10.8,
                    "relative_humidity_2m": 68,
                    "is_day": 0,
                    "precipitation": 0.2,
                    "rain": 0.2,
                    "snowfall": 0.0,
                    "visibility": 9500.0,
                    "weather_code": 61,
                    "wind_speed_10m": 14.5,
                    "wind_gusts_10m": 24.0
                }
            }

    monkeypatch.setattr("src.meteorologia.requests.get", lambda *args, **kwargs: MockResponse())
    df_meteo = obtener_meteorologia_actual()

    assert len(df_meteo) == 1
    assert df_meteo.shape[1] == 13
    assert float(df_meteo["temperatura_c"].iloc[0]) == 12.4
    assert float(df_meteo["visibilidad_m"].iloc[0]) == 9500.0
    assert int(df_meteo["es_de_dia"].iloc[0]) == 0


def test_emt_rotacion_estratificada_y_control_cuota(monkeypatch):
    """Verifica el intercalado M-30/Periferia y el bloqueo autónomo al alcanzar el límite."""
    secuencia = construir_lista_paradas_estratificada()
    assert len(secuencia) == 4928
    assert len(set(secuencia)) == 4928

    monkeypatch.setenv("EMT_CLIENT_ID", "test_id")
    monkeypatch.setenv("EMT_PASSKEY", "test_key")

    cliente = ClienteEMTRealtime(limite_ciclo=2)
    cliente.estado = {
        "session_id": "SES_TEST",
        "session_start_utc": "2026-11-18T18:00:00+00:00",
        "last_request_utc": None,
        "next_allowed_start_utc": None,
        "peticiones_realizadas": 0,
        "indice_parada_actual": 0,
        "en_pausa_cuota": False
    }

    monkeypatch.setattr("src.emt_realtime._guardar_estado_cuota", lambda estado, **kw: None)
    monkeypatch.setattr(
        cliente,
        "_consultar_parada",
        lambda stop_id: [{"line": "27", "destination": "PLAZA CASTILLA", "bus": "101", "estimateArrive": 120, "DistanceBus": 300}]
    )

    df_lote, _ = cliente.ejecutar_lote_rotativo(tamano_lote=5)
    assert len(df_lote) == 2
    assert cliente.estado["en_pausa_cuota"] is True
    assert cliente.estado["next_allowed_start_utc"] is not None

    df_vacio, estado_pausa = cliente.ejecutar_lote_rotativo(tamano_lote=5)
    assert df_vacio.empty
    assert "STANDBY_CUOTA" in estado_pausa
