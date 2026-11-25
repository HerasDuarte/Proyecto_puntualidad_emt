import base64
import gzip
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest

import scripts.archivar_nube_a_hf as archivar_hf
import scripts.descargar_historico_hf as descargar_hf
import scripts.generar_catalogo_dashboard as gen_cat
import scripts.sincronizar_desde_cloudflare as sync_cf
from scripts.sincronizar_desde_cloudflare import _parsear_xml_trafico_gzip
from src.emt_realtime import ClienteEMTRealtime, _cargar_estado_cuota, construir_lista_paradas_estratificada
from src.limpieza import limpiar_frequencies, limpiar_routes
from src.meteorologia import obtener_meteorologia_actual
from src.sensores_trafico import sanear_xml_madrid, vincular_trafico_a_paradas
from src.utils import (
    calcular_distancia_haversine,
    extraer_contexto_madrid,
    firmar_grafico_emt,
    formatear_hora_madrid,
    parsear_xml_trafico_gzip,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

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
    """Verifica que las cuarentenas tras actualizaciones GTFS sean inexistentes o estrictamente residuales (<1%)."""
    processed_dir = ROOT_DIR / 'data' / 'processed'
    cuarentena_dir = processed_dir / 'cuarentena'
    cuarentenas = list(cuarentena_dir.glob('quarantine_*.csv')) if cuarentena_dir.exists() else list()
    total_maestro = len(pd.read_parquet(processed_dir / 'maestro_gtfs.parquet', columns=['trip_id']))

    for archivo_q in cuarentenas:
        df_q = pd.read_csv(archivo_q)
        tasa_q = len(df_q) / max(total_maestro, 1)
        assert tasa_q < 0.01, f"Tasa de cuarentena excesiva ({tasa_q:.2%}) en {archivo_q.name}"


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
    assert df_res.loc[0, "trafico_regimen"] in {"Fluido", "Saturado_Con_Movimiento", "Colapso_Estatico", "Libre"}
    assert not np.isnan(float(df_res.loc[0, "trafico_indice_friccion"]))
    assert not np.isnan(float(df_res.loc[0, "trafico_presion_entorno"]))

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
    df_stops_vigente = pd.read_csv(ROOT_DIR / 'data' / 'processed' / 'stops_limpio.csv')
    n_paradas_activas = len(df_stops_vigente)

    secuencia = construir_lista_paradas_estratificada()
    assert n_paradas_activas >= 4900
    assert len(secuencia) == n_paradas_activas
    assert len(set(secuencia)) == n_paradas_activas

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

def test_reinicio_diario_cuota_utc(tmp_path):
    """Verifica que el cambio de fecha UTC reinicia peticiones y pausa conservando el índice."""
    ruta_estado = tmp_path / "estado_cuota_emt.json"
    estado_ayer = {
        "fecha_cuota_utc": "2020-01-01",
        "peticiones_realizadas": 4500,
        "indice_parada_actual": 1452,
        "en_pausa_cuota": True,
    }
    ruta_estado.write_text(json.dumps(estado_ayer), encoding="utf-8")

    estado_hoy = _cargar_estado_cuota(ruta_estado)
    hoy_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    assert estado_hoy["fecha_cuota_utc"] == hoy_utc
    assert estado_hoy["peticiones_realizadas"] == 0
    assert estado_hoy["en_pausa_cuota"] is False
    assert estado_hoy["indice_parada_actual"] == 1452


def test_parsear_xml_trafico_gzip_sanea_y_filtra():
    """Valida descompresión GZIP, saneamiento de ampersand suelto y filtrado de sensores averiados."""
    xml_raw = """<?xml version="1.0" encoding="UTF-8"?>
    <pms>
      <fecha_hora>20/11/2026 10:00:00</fecha_hora>
      <pm>
        <idelem>1001</idelem>
        <descripcion>Castellana & Goya</descripcion>
        <tipo_elem>URB</tipo_elem>
        <intensidad>520</intensidad>
        <ocupacion>14</ocupacion>
        <carga>32</carga>
        <nivelServicio>1</nivelServicio>
        <velocidad>45</velocidad>
        <error>N</error>
        <st_x>441234,5</st_x>
        <st_y>4475123,2</st_y>
      </pm>
      <pm>
        <idelem>9999</idelem>
        <descripcion>Sensor Averiado</descripcion>
        <error>S</error>
      </pm>
    </pms>"""
    xml_gzip_b64 = base64.b64encode(gzip.compress(xml_raw.encode("utf-8"))).decode("ascii")

    df = _parsear_xml_trafico_gzip(xml_gzip_b64, "2026-11-20T09:00:00Z")

    assert len(df) == 1
    assert df.iloc[0]["idelem"] == "1001"
    assert df.iloc[0]["intensidad"] == 520
    assert df.iloc[0]["st_x"] == pytest.approx(441234.5)


def test_firmar_grafico_emt_anade_atribucion():
    """Comprueba que firmar_grafico_emt inserta el texto legal requerido por MobilityLabs."""
    fig, _ = plt.subplots()
    firmar_grafico_emt(fig)
    textos = [t.get_text() for t in fig.texts]
    plt.close(fig)

    assert any("Powered by EMT de Madrid" in txt for txt in textos)

def _crear_zip_gtfs_sintetico(contenido_calendar: str = "S1,1,1,1,1,1,0,0,20260101,20261231", contenido_agency: str = "EMT,Madrid") -> bytes:
    """Construye en memoria un archivo ZIP GTFS sintético con las 8 tablas operativas + agency.txt."""
    buffer = io.BytesIO()
    tablas = {
        "agency.txt": contenido_agency,
        "feed_info.txt": "EMT,http://emtmadrid.es,es",
        "routes.txt": "001,1,Cristo Rey - Prosperidad,3",
        "stops.txt": "100,Cibeles,40.419,-3.693",
        "calendar.txt": contenido_calendar,
        "calendar_dates.txt": "S1,20261225,2",
        "trips.txt": "001,S1,T1,Prosperidad,0,B1,SH1",
        "frequencies.txt": "T1,07:00:00,22:00:00,600",
        "shapes.txt": "SH1,40.419,-3.693,1,0",
        "stop_times.txt": "T1,07:00:00,07:00:00,100,1,0",
    }
    with zipfile.ZipFile(buffer, "w") as zf:
        for nombre, texto in tablas.items():
            zf.writestr(nombre, texto.encode("utf-8"))
    return buffer.getvalue()


def test_calcular_hashes_operativos_excluye_agency_y_feed_info():
    """Verifica que solo se auditan las 8 tablas operativas y se excluyen agency.txt y feed_info.txt."""
    zip_bytes = _crear_zip_gtfs_sintetico()
    hashes = sync_cf._calcular_hashes_operativos_zip(zip_bytes)

    assert len(hashes) == 8
    assert set(hashes.keys()) == set(sync_cf.TABLAS_GTFS_OPERATIVAS)
    assert "agency.txt" not in hashes
    assert "feed_info.txt" not in hashes

def test_auditoria_gtfs_ignora_cambios_en_agency(tmp_path, monkeypatch):
    """Comprueba que cambiar solo agency.txt no dispara reconstrucción del maestro GTFS."""
    dir_raw = tmp_path / "raw"
    dir_proc = tmp_path / "processed"
    dir_cuar = dir_proc / "cuarentena"
    dir_vers = dir_proc / "versiones_gtfs"
    (dir_raw / "gtfs").mkdir(parents=True)
    dir_proc.mkdir(parents=True)

    zip_v1 = _crear_zip_gtfs_sintetico(contenido_agency="EMT,Telefono_Antiguo")
    with zipfile.ZipFile(io.BytesIO(zip_v1), "r") as zf:
        for nombre in sync_cf.TABLAS_GTFS_OPERATIVAS:
            (dir_raw / "gtfs" / nombre).write_bytes(zf.read(nombre))

    monkeypatch.setattr(sync_cf, "DIR_RAW", dir_raw)
    monkeypatch.setattr(sync_cf, "DIR_PROCESSED", dir_proc)
    monkeypatch.setattr(sync_cf, "DIR_CUARENTENA", dir_cuar)
    monkeypatch.setattr(sync_cf, "DIR_VERSIONES_GTFS", dir_vers)
    monkeypatch.setattr(sync_cf, "RUTA_HISTORIAL_GTFS", dir_vers / "historial_versiones_gtfs.json")

    zip_v2 = _crear_zip_gtfs_sintetico(contenido_agency="EMT,Telefono_Nuevo_900")
    with patch.object(sync_cf, "_descargar_zip_gtfs_oficial", return_value=zip_v2):
        hubo_cambio = sync_cf.auditar_y_actualizar_gtfs_diario()

    assert hubo_cambio is False
    assert (dir_vers / "historial_versiones_gtfs.json").exists()

def test_auditoria_gtfs_reconstruye_ante_cambio_en_calendar(tmp_path, monkeypatch):
    """Valida respaldo previo, archivado versionado y reconstrucción al cambiar calendar.txt."""
    dir_raw = tmp_path / "raw"
    dir_proc = tmp_path / "processed"
    dir_cuar = dir_proc / "cuarentena"
    dir_vers = dir_proc / "versiones_gtfs"
    (dir_raw / "gtfs").mkdir(parents=True)
    dir_proc.mkdir(parents=True)

    zip_v1 = _crear_zip_gtfs_sintetico(contenido_calendar="S1,1,1,1,1,1,0,0,20260101,20261220")
    with zipfile.ZipFile(io.BytesIO(zip_v1), "r") as zf:
        for nombre in sync_cf.TABLAS_GTFS_OPERATIVAS:
            (dir_raw / "gtfs" / nombre).write_bytes(zf.read(nombre))

    maestro_inicial = dir_proc / "maestro_gtfs.parquet"
    maestro_inicial.write_bytes(b"MAESTRO_V1_BYTES")

    monkeypatch.setattr(sync_cf, "DIR_RAW", dir_raw)
    monkeypatch.setattr(sync_cf, "DIR_PROCESSED", dir_proc)
    monkeypatch.setattr(sync_cf, "DIR_CUARENTENA", dir_cuar)
    monkeypatch.setattr(sync_cf, "DIR_VERSIONES_GTFS", dir_vers)
    monkeypatch.setattr(sync_cf, "RUTA_HISTORIAL_GTFS", dir_vers / "historial_versiones_gtfs.json")

    zip_navidad = _crear_zip_gtfs_sintetico(contenido_calendar="S1,1,1,1,1,1,0,0,20261221,20270107")

    def _simular_pipeline(cmd, check=True):
        if "construir_maestro_gtfs.py" in cmd[-1]:
            maestro_inicial.write_bytes(b"MAESTRO_NAVIDAD_V2_BYTES")

    with patch.object(sync_cf, "_descargar_zip_gtfs_oficial", return_value=zip_navidad), \
         patch.object(sync_cf.subprocess, "run", side_effect=_simular_pipeline):
        hubo_cambio = sync_cf.auditar_y_actualizar_gtfs_diario()

    assert hubo_cambio is True
    respaldos = list(dir_vers.glob("maestro_gtfs_previo_*.parquet"))
    versionados = list(dir_vers.glob("maestro_gtfs_20*.parquet"))
    assert len(respaldos) == 1
    assert next(iter(respaldos)).read_bytes() == b"MAESTRO_V1_BYTES"
    assert len(versionados) == 1
    assert next(iter(versionados)).read_bytes() == b"MAESTRO_NAVIDAD_V2_BYTES"

def test_anexar_parquet_evita_duplicados(tmp_path):
    """Comprueba que _anexar_parquet deduplica registros repetidos por clave compuesta."""
    ruta_pq = tmp_path / "meteo_test.parquet"
    df1 = pd.DataFrame([
        {"timestamp_captura_utc": "2026-11-20T10:00:00Z", "temperatura_c": 12.5},
        {"timestamp_captura_utc": "2026-11-20T10:15:00Z", "temperatura_c": 13.0},
    ])
    df2 = pd.DataFrame([
        {"timestamp_captura_utc": "2026-11-20T10:15:00Z", "temperatura_c": 13.1},
        {"timestamp_captura_utc": "2026-11-20T10:30:00Z", "temperatura_c": 13.4},
    ])

    sync_cf._anexar_parquet(df1, ruta_pq, ["timestamp_captura_utc"])
    total = sync_cf._anexar_parquet(df2, ruta_pq, ["timestamp_captura_utc"])

    assert total == 3
    df_final = pd.read_parquet(ruta_pq)
    assert float(df_final.loc[df_final["timestamp_captura_utc"] == "2026-11-20T10:15:00Z", "temperatura_c"].iloc[0]) == 13.1

def test_reconciliar_cuarentenas_gtfs_rescate_y_purga(tmp_path, monkeypatch):
    """Verifica que reconciliar_cuarentenas_gtfs elimina archivos obsoletos al rescatarse los registros."""
    dir_proc = tmp_path / "processed"
    dir_cuar = dir_proc / "cuarentena"
    dir_cuar.mkdir(parents=True)
    monkeypatch.setattr(sync_cf, "DIR_PROCESSED", dir_proc)
    monkeypatch.setattr(sync_cf, "DIR_CUARENTENA", dir_cuar)

    q_trips = dir_cuar / "quarantine_trips.csv"
    pd.DataFrame([
        {"trip_id": "FE8420011", "route_id": "842"},
        {"trip_id": "FE8420012", "route_id": "842"},
    ]).to_csv(q_trips, index=False)

    q_freqs = dir_cuar / "quarantine_frequencies.csv"
    pd.DataFrame([
        {"trip_id": "FE8420011", "headway_secs": 300},
        {"trip_id": "T_ROTO_999", "headway_secs": 600},
    ]).to_csv(q_freqs, index=False)

    claves_previas = sync_cf._leer_claves_cuarentena()

    ruta_maestro = dir_proc / "maestro_gtfs.parquet"
    pd.DataFrame([
        {"trip_id": "FE8420011", "route_id": "842"},
        {"trip_id": "FE8420012", "route_id": "842"},
    ]).to_parquet(ruta_maestro, index=False)

    rescatados = sync_cf.reconciliar_cuarentenas_gtfs(claves_previas, ruta_maestro)

    assert rescatados == 3
    assert not q_trips.exists()
    assert q_freqs.exists()
    df_restante = pd.read_csv(q_freqs)
    assert list(df_restante["trip_id"]) == ["T_ROTO_999"]

def test_archivar_tabla_d1_a_hf_flujo_transaccional(tmp_path):
    """Verifica fusión deduplicada, commit en Hugging Face y purga posterior en Cloudflare D1."""
    mock_api = MagicMock()
    mock_api.token = "hf_test_token"

    # Simular que ya existía un registro previo en Hugging Face
    pq_remoto_previo = tmp_path / "meteo_madrid_20261121.parquet"
    pd.DataFrame([
        {"timestamp_captura_utc": "2026-11-21T10:00:00Z", "temperatura_c": 11.2},
    ]).to_parquet(pq_remoto_previo, index=False)

    # Simular 2 lotes en Cloudflare D1: primero 2 filas nuevas (IDs 41 y 42) y luego vacío
    resp_export_1 = MagicMock()
    resp_export_1.json.return_value = {
        "rows": [
            {"id": 41, "timestamp_captura_utc": "2026-11-21T10:00:00Z", "temperatura_c": 11.3},
            {"id": 42, "timestamp_captura_utc": "2026-11-21T10:15:00Z", "temperatura_c": 11.8},
        ]
    }
    resp_export_vacio = MagicMock()
    resp_export_vacio.json.return_value = {"rows": list()}
    resp_purge = MagicMock()

    with patch.object(archivar_hf, "hf_hub_download", return_value=str(pq_remoto_previo)), \
         patch.object(archivar_hf.requests, "get", side_effect=[resp_export_1, resp_export_vacio]), \
         patch.object(archivar_hf.requests, "post", return_value=resp_purge) as mock_post:
        total = archivar_hf.archivar_tabla_d1_a_hf(
            mock_api,
            "HerasDuarte/emt-madrid-realtime-data",
            "https://worker.test",
            "secret_token",
            "meteo_madrid",
        )

    assert total == 2
    mock_api.create_commit.assert_called_once()
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"] == {"table": "meteo_madrid", "max_id": 42}

def test_archivar_tabla_d1_a_hf_no_purga_si_falla_hf():
    """Garantiza que nunca se purga Cloudflare D1 si falla la subida a Hugging Face."""
    mock_api = MagicMock()
    mock_api.token = "hf_test_token"
    mock_api.create_commit.side_effect = RuntimeError("Error simulado de red en Hugging Face")

    resp_export = MagicMock()
    resp_export.json.return_value = {
        "rows": [
            {"id": 99, "timestamp_captura_utc": "2026-11-21T12:00:00Z", "temperatura_c": 14.0},
        ]
    }

    with patch.object(archivar_hf, "_descargar_fichero_remoto_si_existe", return_value=False), \
         patch.object(archivar_hf.requests, "get", return_value=resp_export), \
         patch.object(archivar_hf.requests, "post") as mock_post:
        with pytest.raises(RuntimeError, match="Error simulado"):
            archivar_hf.archivar_tabla_d1_a_hf(
                mock_api,
                "HerasDuarte/emt-madrid-realtime-data",
                "https://worker.test",
                "secret_token",
                "meteo_madrid",
            )

    mock_post.assert_not_called()


def test_respaldar_snapshot_gtfs_en_hf_detecta_cambios(tmp_path):
    """Verifica que respaldar_snapshot_gtfs_en_hf archiva el ZIP en HF solo cuando cambian las 8 tablas."""
    mock_api = MagicMock()
    mock_api.token = "hf_test_token"

    zip_v1 = _crear_zip_gtfs_sintetico(contenido_calendar="S1,1,1,1,1,1,0,0,20260101,20261220")
    hashes_v1 = archivar_hf._calcular_hashes_operativos_zip(zip_v1)

    hist_file = tmp_path / "historial_gtfs_nube.json"
    hist_file.write_text(
        json.dumps([{"version_tag": "20261120", "hashes_tablas": hashes_v1}]),
        encoding="utf-8",
    )

    # Caso 1: Sin cambios en las 8 tablas (aunque cambie agency.txt) -> no sube nada a HF
    zip_solo_agency = _crear_zip_gtfs_sintetico(
        contenido_calendar="S1,1,1,1,1,1,0,0,20260101,20261220",
        contenido_agency="EMT,Nuevo_Telefono_900",
    )
    with patch.object(archivar_hf, "_descargar_zip_gtfs_oficial", return_value=zip_solo_agency), \
         patch.object(archivar_hf, "hf_hub_download", return_value=str(hist_file)):
        subido_1 = archivar_hf.respaldar_snapshot_gtfs_en_hf(mock_api, "HerasDuarte/emt-madrid-realtime-data")

    assert subido_1 is False
    mock_api.create_commit.assert_not_called()

    # Caso 2: Cambio real en calendar.txt (ej. horario de Navidad) -> archiva ZIP + historial en HF
    zip_navidad = _crear_zip_gtfs_sintetico(contenido_calendar="S1,1,1,1,1,1,0,0,20261221,20270107")
    with patch.object(archivar_hf, "_descargar_zip_gtfs_oficial", return_value=zip_navidad), \
         patch.object(archivar_hf, "hf_hub_download", return_value=str(hist_file)):
        subido_2 = archivar_hf.respaldar_snapshot_gtfs_en_hf(mock_api, "HerasDuarte/emt-madrid-realtime-data")

    assert subido_2 is True
    mock_api.create_commit.assert_called_once()
    rutas_subidas = [op.path_in_repo for op in mock_api.create_commit.call_args.kwargs["operations"]]
    assert any(r.startswith("gtfs_snapshots/gtfs_") and r.endswith(".zip") for r in rutas_subidas)
    assert "gtfs_snapshots/historial_gtfs_nube.json" in rutas_subidas

def test_descargar_historico_desde_hf_incremental(tmp_path, monkeypatch):
    """Verifica descarga incremental desde Hugging Face evitando redescargar históricos ya presentes."""
    dir_raw = tmp_path / "raw"
    dir_vers_raw = dir_raw / "versiones_gtfs"
    (dir_raw / "realtime" / "emt").mkdir(parents=True)
    dir_vers_raw.mkdir(parents=True)

    # Simular que el día antiguo y el ZIP previo ya existen en disco local
    (dir_raw / "realtime" / "emt" / "emt_llegadas_20261118.parquet").write_bytes(b"PARQUET_ANTIGUO_OK")
    (dir_vers_raw / "gtfs_20261120.zip").write_bytes(b"ZIP_ANTIGUO_OK")

    monkeypatch.setattr(descargar_hf, "DIR_RAW", dir_raw)
    monkeypatch.setattr(descargar_hf, "DIR_VERSIONES_GTFS_RAW", dir_vers_raw)

    mock_api = MagicMock()
    mock_api.token = "hf_test_token"
    mock_api.list_repo_files.return_value = [
        "README.md",
        "realtime/emt/emt_llegadas_20261118.parquet",
        "realtime/emt/emt_llegadas_20261121.parquet",
        "gtfs_snapshots/gtfs_20261120.zip",
        "gtfs_snapshots/historial_gtfs_nube.json",
    ]

    cache_dir = tmp_path / "hf_cache"
    cache_dir.mkdir()

    def _simular_descarga_hf(repo_id, filename, repo_type, token):
        destino_cache = cache_dir / Path(filename).name
        destino_cache.write_bytes(f"CONTENIDO_{Path(filename).name}".encode("utf-8"))
        return str(destino_cache)

    with patch.object(descargar_hf, "_obtener_cliente_hf", return_value=(mock_api, "HerasDuarte/emt-madrid-realtime-data")), \
         patch.object(descargar_hf, "hf_hub_download", side_effect=_simular_descarga_hf):
        resumen = descargar_hf.descargar_historico_desde_hf(forzar_actualizacion=False)

    # Solo debe descargar el Parquet del día más reciente (20261121) y refrescar el JSON de historial
    assert resumen["parquets_actualizados"] == 1
    assert resumen["snapshots_gtfs_actualizados"] == 1
    assert (dir_raw / "realtime" / "emt" / "emt_llegadas_20261118.parquet").read_bytes() == b"PARQUET_ANTIGUO_OK"
    assert (dir_raw / "realtime" / "emt" / "emt_llegadas_20261121.parquet").read_bytes() == b"CONTENIDO_emt_llegadas_20261121.parquet"
    assert (dir_vers_raw / "historial_gtfs_nube.json").exists()


# ==============================================================================
# 4. PRUEBAS DE UTILIDADES CENTRALIZADAS, VALIDACIONES Y CATÁLOGO DASHBOARD
# ==============================================================================


def test_utilidades_horario_madrid_y_haversine():
    """Valida formateo CET/CEST, clasificación LA/SA/FE y Haversine."""
    assert formatear_hora_madrid(None) == "Sin datos"
    assert formatear_hora_madrid("2026-11-24T12:00:00Z") == "24/11/2026 13:00:00"
    assert formatear_hora_madrid("2026-07-15T12:00:00Z") == "15/07/2026 14:00:00"
    assert formatear_hora_madrid("fecha-invalida") == "fecha-invalida"

    # Martes 24/11/2026 -> LA | Sábado 28/11/2026 -> SA | Domingo 29/11/2026 -> FE
    assert extraer_contexto_madrid("2026-11-24T12:00:00Z") == ("LA", 13)
    assert extraer_contexto_madrid("2026-11-28T12:00:00Z") == ("SA", 13)
    assert extraer_contexto_madrid("2026-11-29T12:00:00Z") == ("FE", 13)

    assert calcular_distancia_haversine(40.4168, -3.7038, 40.4168, -3.7038) == pytest.approx(0.0)
    dists = calcular_distancia_haversine(
        40.4192, -3.6931, np.array([40.4168, 40.4192]), np.array([-3.7038, -3.6931])
    )
    assert 800.0 < float(dists[0]) < 1100.0
    assert float(dists[1]) == pytest.approx(0.0)


def test_trafico_filtra_porcentajes_fuera_rango_y_xml_corrupto():
    """Verifica saneamiento >100%, XML corrupto seguro, timestamp_trafico y DataFrames vacíos."""
    xml_anomalo = """<?xml version="1.0" encoding="UTF-8"?>
    <pms>
      <fecha_hora>25/11/2026 11:00:00</fecha_hora>
      <pm>
        <idelem>PM43012</idelem>
        <descripcion>Sensor anomalo</descripcion>
        <tipo_elem>URB</tipo_elem>
        <intensidad>-10</intensidad>
        <ocupacion>150</ocupacion>
        <carga>189</carga>
        <nivelServicio>2</nivelServicio>
        <velocidad>40</velocidad>
        <error>N</error>
        <st_x>441000</st_x>
        <st_y>4475000</st_y>
      </pm>
    </pms>"""
    xml_b64 = base64.b64encode(gzip.compress(xml_anomalo.encode("utf-8"))).decode("ascii")
    df_sano = parsear_xml_trafico_gzip(xml_b64, "2026-11-25T10:00:00Z")

    assert len(df_sano) == 1
    assert np.isnan(df_sano.iloc[0]["intensidad"])
    assert np.isnan(df_sano.iloc[0]["ocupacion"])
    assert np.isnan(df_sano.iloc[0]["carga"])
    assert float(df_sano.iloc[0]["nivelServicio"]) == 2.0

    # XML corrupto o vacío debe devolver DataFrame vacío sin abortar el flujo
    assert parsear_xml_trafico_gzip("", "2026-11-25T10:00:00Z").empty
    corrupto_b64 = base64.b64encode(b"NO_ES_GZIP").decode("ascii")
    assert parsear_xml_trafico_gzip(corrupto_b64, "2026-11-25T10:00:00Z").empty

    # Validar que timestamp_trafico en vincular_trafico_a_paradas es texto real y tolera vacíos
    df_stops = pd.DataFrame({"stop_id": ["923"], "stop_lat": [40.42], "stop_lon": [-3.69]})
    df_sens = pd.DataFrame(
        {
            "timestamp_trafico": ["25/11/2026 11:00:00"],
            "idelem": ["S1"],
            "carga": [45.0],
            "ocupacion": [20.0],
            "intensidad": [600.0],
            "nivel_servicio": [1.0],
            "lat": [40.4205],
            "lon": [-3.6905],
        }
    )
    df_vinc = vincular_trafico_a_paradas(df_stops, df_sens)
    assert isinstance(df_vinc.loc[0, "timestamp_trafico"], str)
    assert df_vinc.loc[0, "timestamp_trafico"] == "25/11/2026 11:00:00"

    df_vacio = vincular_trafico_a_paradas(df_stops, df_sens.iloc[0:0])
    assert df_vacio.empty
    assert "timestamp_trafico" in df_vacio.columns


def test_errores_red_meteorologia_y_autenticacion_emt(monkeypatch, tmp_path):
    """Valida errores descriptivos ante respuestas incompletas de Open-Meteo y EMT."""
    class RespSinCurrent:
        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "ok"}

    monkeypatch.setattr(
        "src.meteorologia.requests.get", lambda *args, **kwargs: RespSinCurrent()
    )
    with pytest.raises(ValueError, match="current"):
        obtener_meteorologia_actual()

    ruta_stops = tmp_path / "stops_limpio.csv"
    pd.DataFrame({"stop_id": ["923"], "en_alta_densidad_real": [True]}).to_csv(
        ruta_stops, index=False
    )
    monkeypatch.setenv("EMT_CLIENT_ID", "id_test")
    monkeypatch.setenv("EMT_PASSKEY", "key_test")
    monkeypatch.setattr("src.emt_realtime.RUTA_STOPS_PROCESADO", ruta_stops)
    monkeypatch.setattr("src.emt_realtime.RUTA_ESTADO_CUOTA", tmp_path / "cuota.json")

    class RespSinToken:
        def raise_for_status(self):
            pass

        def json(self):
            return {"code": "00", "data": [{}]}

    monkeypatch.setattr(
        "src.emt_realtime.requests.get", lambda *args, **kwargs: RespSinToken()
    )
    cliente = ClienteEMTRealtime(limite_ciclo=10)
    assert cliente.paradas_rotativas == ["923"]
    with pytest.raises(RuntimeError, match="accessToken"):
        cliente._autenticar()


def test_generar_catalogo_dashboard_completo(tmp_path):
    """Verifica la generación de los 5 Parquet ligeros y sus cálculos derivados."""
    dir_gtfs = tmp_path / "gtfs"
    dir_out = tmp_path / "catalogo_out"
    dir_gtfs.mkdir()

    (dir_gtfs / "routes.txt").write_text(
        "route_id,agency_id,route_short_name,route_long_name,route_color\n"
        "023,EMT,23,Plaza Mayor - Villaverde,0178BC\n",
        encoding="utf-8",
    )
    (dir_gtfs / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\n"
        "923,Plaza Mayor,40.415,-3.707\n"
        "924,Atocha,40.406,-3.691\n",
        encoding="utf-8",
    )
    (dir_gtfs / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday\n"
        "LA,1,1,1,1,1,0,0\n",
        encoding="utf-8",
    )
    (dir_gtfs / "trips.txt").write_text(
        "route_id,service_id,trip_id,trip_headsign,direction_id,shape_id\n"
        "023,LA,T23_0,Villaverde,0,023_A\n",
        encoding="utf-8",
    )
    (dir_gtfs / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled\n"
        "T23_0,08:00:00,08:00:00,923,1,0\n"
        "T23_0,08:10:00,08:10:00,924,2,2500.5\n",
        encoding="utf-8",
    )
    (dir_gtfs / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
        "023_A,40.415,-3.707,1\n"
        "023_A,40.406,-3.691,2\n",
        encoding="utf-8",
    )
    (dir_gtfs / "frequencies.txt").write_text(
        "trip_id,start_time,end_time,headway_secs\n"
        "T23_0,08:00:00,09:59:59,600\n",
        encoding="utf-8",
    )

    archivos = gen_cat.construir_catalogo_dashboard(
        dir_gtfs=dir_gtfs, dir_salida=dir_out
    )

    assert len(archivos) == 5
    for ruta_pq in archivos.values():
        assert ruta_pq.exists()

    df_par = pd.read_parquet(archivos["paradas"])
    assert df_par.loc[df_par["stop_id"] == "923", "lineas"].iloc[0] == "23"

    df_rec = pd.read_parquet(archivos["recorridos_paradas"])
    assert float(df_rec.loc[df_rec["stop_id"] == "923", "min_desde_cabecera"].iloc[0]) == 0.0
    assert float(df_rec.loc[df_rec["stop_id"] == "924", "min_desde_cabecera"].iloc[0]) == 10.0

    df_freq = pd.read_parquet(archivos["frecuencias_oficiales"])
    assert set(df_freq["hora"].tolist()) == {8, 9}
    assert float(df_freq["headway_min"].iloc[0]) == 10.0


def test_archivar_nube_envia_token_en_cabecera_y_tolera_xml_corrupto():
    """Valida envío de X-Sync-Token en headers y tolerancia a XML corrupto en tráfico."""
    mock_api = MagicMock()
    mock_api.token = "hf_test_token"

    resp_export_1 = MagicMock()
    resp_export_1.json.return_value = {
        "rows": [
            {
                "id": 77,
                "timestamp_captura_utc": "2026-11-25T10:00:00Z",
                "xml_gzip_base64": "NO_VALIDO",
            }
        ]
    }
    resp_export_vacio = MagicMock()
    resp_export_vacio.json.return_value = {"rows": list()}
    resp_purge = MagicMock()

    with patch.object(
        archivar_hf.requests, "get", side_effect=[resp_export_1, resp_export_vacio]
    ) as mock_get, patch.object(
        archivar_hf.requests, "post", return_value=resp_purge
    ) as mock_post:
        total = archivar_hf.archivar_tabla_d1_a_hf(
            mock_api,
            "HerasDuarte/emt-madrid-realtime-data",
            "https://worker.test",
            "mi_secreto_sync",
            "trafico_xml_raw",
        )

    assert total == 1
    assert mock_get.call_args_list[0].kwargs["headers"]["X-Sync-Token"] == "mi_secreto_sync"
    assert "token" not in mock_get.call_args_list[0].kwargs["params"]
    assert mock_post.call_args.kwargs["headers"]["X-Sync-Token"] == "mi_secreto_sync"
    assert "params" not in mock_post.call_args.kwargs
