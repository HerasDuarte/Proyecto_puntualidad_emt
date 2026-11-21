import pytest
import pandas as pd
import polars as pl
from pathlib import Path
import sys
import numpy as np
import base64
import gzip
import json
from datetime import datetime, timezone
import matplotlib.pyplot as plt

import io
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.sincronizar_desde_cloudflare as sync_cf

from src.emt_realtime import _cargar_estado_cuota
from src.utils import firmar_grafico_emt
from scripts.sincronizar_desde_cloudflare import _parsear_xml_trafico_gzip

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
    """Verifica que las cuarentenas tras actualizaciones GTFS sean inexistentes o estrictamente residuales (<1%)."""
    processed_dir = ROOT_DIR / 'data' / 'processed'
    cuarentenas = list(processed_dir.glob('quarantine_*.csv'))
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
    (dir_raw / "gtfs").mkdir(parents=True)
    dir_proc.mkdir(parents=True)

    zip_v1 = _crear_zip_gtfs_sintetico(contenido_agency="EMT,Telefono_Antiguo")
    with zipfile.ZipFile(io.BytesIO(zip_v1), "r") as zf:
        for nombre in sync_cf.TABLAS_GTFS_OPERATIVAS:
            (dir_raw / "gtfs" / nombre).write_bytes(zf.read(nombre))

    monkeypatch.setattr(sync_cf, "DIR_RAW", dir_raw)
    monkeypatch.setattr(sync_cf, "DIR_PROCESSED", dir_proc)
    monkeypatch.setattr(sync_cf, "RUTA_HISTORIAL_GTFS", dir_proc / "historial_versiones_gtfs.json")

    zip_v2 = _crear_zip_gtfs_sintetico(contenido_agency="EMT,Telefono_Nuevo_900")
    with patch.object(sync_cf, "_descargar_zip_gtfs_oficial", return_value=zip_v2):
        hubo_cambio = sync_cf.auditar_y_actualizar_gtfs_diario()

    assert hubo_cambio is False
    assert (dir_proc / "historial_versiones_gtfs.json").exists()


def test_auditoria_gtfs_reconstruye_ante_cambio_en_calendar(tmp_path, monkeypatch):
    """Valida respaldo previo, archivado versionado y reconstrucción al cambiar calendar.txt."""
    dir_raw = tmp_path / "raw"
    dir_proc = tmp_path / "processed"
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
    monkeypatch.setattr(sync_cf, "RUTA_HISTORIAL_GTFS", dir_proc / "historial_versiones_gtfs.json")

    zip_navidad = _crear_zip_gtfs_sintetico(contenido_calendar="S1,1,1,1,1,1,0,0,20261221,20270107")

    def _simular_pipeline(cmd, check=True):
        if "construir_maestro_gtfs.py" in cmd[-1]:
            maestro_inicial.write_bytes(b"MAESTRO_NAVIDAD_V2_BYTES")

    with patch.object(sync_cf, "_descargar_zip_gtfs_oficial", return_value=zip_navidad), \
         patch.object(sync_cf.subprocess, "run", side_effect=_simular_pipeline):
        hubo_cambio = sync_cf.auditar_y_actualizar_gtfs_diario()

    assert hubo_cambio is True
    respaldos = list(dir_proc.glob("maestro_gtfs_previo_*.parquet"))
    versionados = list(dir_proc.glob("maestro_gtfs_20*.parquet"))
    assert len(respaldos) == 1
    assert respaldos[0].read_bytes() == b"MAESTRO_V1_BYTES"
    assert len(versionados) == 1
    assert versionados[0].read_bytes() == b"MAESTRO_NAVIDAD_V2_BYTES"


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
    dir_proc.mkdir(parents=True)
    monkeypatch.setattr(sync_cf, "DIR_PROCESSED", dir_proc)

    # 1. Crear dos archivos de cuarentena previos: uno totalmente rescatable y otro parcialmente huérfano
    q_trips = dir_proc / "quarantine_trips.csv"
    pd.DataFrame([
        {"trip_id": "FE8420011", "route_id": "842"},
        {"trip_id": "FE8420012", "route_id": "842"},
    ]).to_csv(q_trips, index=False)

    q_freqs = dir_proc / "quarantine_frequencies.csv"
    pd.DataFrame([
        {"trip_id": "FE8420011", "headway_secs": 300},
        {"trip_id": "T_ROTO_999", "headway_secs": 600},
    ]).to_csv(q_freqs, index=False)

    claves_previas = sync_cf._leer_claves_cuarentena()

    # 2. Simular nuevo maestro_gtfs.parquet donde la línea 842 ya fue completada por EMT
    ruta_maestro = dir_proc / "maestro_gtfs.parquet"
    pd.DataFrame([
        {"trip_id": "FE8420011", "route_id": "842"},
        {"trip_id": "FE8420012", "route_id": "842"},
    ]).to_parquet(ruta_maestro, index=False)

    rescatados = sync_cf.reconciliar_cuarentenas_gtfs(claves_previas, ruta_maestro)

    # FE8420011, FE8420012 (2 trips) + 842 (1 route) = 3 claves recuperadas
    assert rescatados == 3
    # quarantine_trips.csv quedó vacío tras el rescate -> debe haber sido eliminado
    assert not q_trips.exists()
    # quarantine_frequencies.csv conserva únicamente el registro que sigue huérfano
    assert q_freqs.exists()
    df_restante = pd.read_csv(q_freqs)
    assert list(df_restante["trip_id"]) == ["T_ROTO_999"]
