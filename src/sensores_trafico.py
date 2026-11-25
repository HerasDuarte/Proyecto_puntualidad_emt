"""
Módulo de Ingesta, Saneamiento y Cruce Espacial de Tráfico Urbano (Ayuntamiento de Madrid).
Descarga el boletín XML en tiempo real (informo.madrid.es), sanea incidencias técnicas
y valores centinela, transforma coordenadas UTM ETRS89 (EPSG:25830) a WGS84 y vincula
métricas ponderadas por inversa de la distancia (IDW) a cada parada de la red EMT.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

from src.utils import (
    HEADERS_HTTP_PROYECTO,
    calcular_distancia_haversine,
    calcular_indice_friccion,
    calcular_regimen_trafico,
    sanear_xml_madrid,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

URL_TRAFICO_OFICIAL = "https://" + "informo." + "madrid.es/informo/tmadrid/pm.xml"
URL_TRAFICO_RESPALDO = (
    "https://"
    + "datos."
    + "madrid.es/egob/catalogo/202087-0-trafico-intensidad.xml"
)


def obtener_trafico_saneado(
    url_xml: str = URL_TRAFICO_OFICIAL, timeout: int = 25
) -> pd.DataFrame:
    """
    Descarga el estado del tráfico en tiempo real del Ayuntamiento de Madrid,
    transforma coordenadas EPSG:25830 a WGS84 (lat/lon), filtra sensores operativos
    (error_sensor == 'N') y convierte valores fuera de rango en NaN.
    """
    headers = {
        "User-Agent": HEADERS_HTTP_PROYECTO["User-Agent"],
        "Accept": "application/xml, text/xml, */*;q=0.9",
    }
    try:
        resp = requests.get(url_xml, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        if url_xml != URL_TRAFICO_OFICIAL:
            raise
        logging.warning(
            "Fallo en URL primaria de tráfico (%s); probando respaldo.", exc
        )
        resp = requests.get(
            URL_TRAFICO_RESPALDO, headers=headers, timeout=timeout
        )
        resp.raise_for_status()

    xml_limpio = sanear_xml_madrid(resp.content)
    try:
        raiz = ET.fromstring(xml_limpio)
    except ET.ParseError as exc:
        raise ValueError(f"El XML de tráfico recibido está truncado o malformado: {exc}") from exc

    fecha_hora_dato = raiz.findtext(
        "fecha_hora",
        default=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    )

    transformador = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)

    registros = []
    for pm in raiz.findall("pm"):
        error_sensor = pm.findtext("error", default="").strip()
        if error_sensor != "N":
            continue

        st_x_raw = pm.findtext("st_x", default="").replace(",", ".")
        st_y_raw = pm.findtext("st_y", default="").replace(",", ".")

        try:
            if not st_x_raw or not st_y_raw:
                continue
            lon, lat = transformador.transform(float(st_x_raw), float(st_y_raw))
        except ValueError:
            continue

        registros.append(
            {
                "timestamp_trafico": fecha_hora_dato,
                "idelem": pm.findtext("idelem"),
                "descripcion": pm.findtext("descripcion", default="").strip(),
                "intensidad": pd.to_numeric(pm.findtext("intensidad"), errors="coerce"),
                "ocupacion": pd.to_numeric(pm.findtext("ocupacion"), errors="coerce"),
                "carga": pd.to_numeric(pm.findtext("carga"), errors="coerce"),
                "nivel_servicio": pd.to_numeric(
                    pm.findtext("nivelServicio"), errors="coerce"
                ),
                "velocidad_trafico": pd.to_numeric(
                    pm.findtext("velocidad"), errors="coerce"
                ),
                "lat": lat,
                "lon": lon,
            }
        )

    df_sensores = pd.DataFrame(registros)
    if df_sensores.empty:
        logging.warning("No se obtuvieron sensores de tráfico operativos.")
        return df_sensores

    # Saneamiento de valores centinela negativos (< 0) y porcentajes > 100 %
    cols_metricas = [
        "intensidad",
        "ocupacion",
        "carga",
        "nivel_servicio",
        "velocidad_trafico",
    ]
    for col in cols_metricas:
        df_sensores.loc[df_sensores[col] < 0, col] = np.nan
    for col_pct in ("ocupacion", "carga"):
        df_sensores.loc[df_sensores[col_pct] > 100, col_pct] = np.nan

    logging.info(
        f"Tráfico saneado ({fecha_hora_dato}): {len(df_sensores):,} sensores operativos listos."
    )
    return df_sensores


def _media_ponderada_idw(
    valores_sensor: np.ndarray, matriz_pesos: np.ndarray
) -> np.ndarray:
    """Calcula el promedio ponderado por inversa de la distancia ignorando valores NaN."""
    valido = ~np.isnan(valores_sensor)[np.newaxis, :]
    w_validos = np.where(valido, matriz_pesos, 0.0)
    suma_pesos = w_validos.sum(axis=1)
    vals_cero = np.nan_to_num(valores_sensor, nan=0.0)
    numerador = w_validos @ vals_cero
    return np.divide(
        numerador,
        suma_pesos,
        out=np.full_like(numerador, np.nan),
        where=suma_pesos > 0,
    )


def vincular_trafico_a_paradas(
    df_stops: pd.DataFrame,
    df_sensores_limpio: pd.DataFrame,
    radio_principal_m: float = 500.0,
    radio_respaldo_m: float = 1000.0,
) -> pd.DataFrame:
    """
    Cruza geodésicamente (Haversine) cada parada EMT con los sensores operativos cercanos
    mediante ponderación IDW en radio principal (500 m) y respaldo (1.000 m), generando
    las 7 variables espaciales de contexto vial para el Modelo 2.
    """
    cols_salida = [
        "stop_id",
        "timestamp_trafico",
        "distancia_sensor_mas_cercano_m",
        "num_sensores_500m",
        "tiene_cobertura_trafico_500m",
        "trafico_carga_entorno",
        "trafico_ocupacion_entorno",
        "trafico_intensidad_entorno",
        "trafico_nivel_servicio_max",
    ]
    if df_stops.empty or df_sensores_limpio.empty:
        return pd.DataFrame(columns=cols_salida)

    lat_stops = df_stops["stop_lat"].to_numpy()[:, np.newaxis]
    lon_stops = df_stops["stop_lon"].to_numpy()[:, np.newaxis]
    lat_sens = df_sensores_limpio["lat"].to_numpy()[np.newaxis, :]
    lon_sens = df_sensores_limpio["lon"].to_numpy()[np.newaxis, :]

    matriz_dist_m = calcular_distancia_haversine(
        lat_stops, lon_stops, lat_sens, lon_sens
    )

    mascara_principal = matriz_dist_m <= radio_principal_m
    mascara_respaldo = matriz_dist_m <= radio_respaldo_m
    tiene_principal = mascara_principal.any(axis=1)

    mascara_efectiva = np.where(
        tiene_principal[:, np.newaxis], mascara_principal, mascara_respaldo
    )

    dist_min_m = matriz_dist_m.min(axis=1)
    num_sensores_500m = mascara_principal.sum(axis=1)
    pesos = np.where(mascara_efectiva, 1.0 / np.maximum(matriz_dist_m, 10.0), 0.0)

    carga_idw = _media_ponderada_idw(df_sensores_limpio["carga"].to_numpy(), pesos)
    ocupacion_idw = _media_ponderada_idw(
        df_sensores_limpio["ocupacion"].to_numpy(), pesos
    )
    intensidad_idw = _media_ponderada_idw(
        df_sensores_limpio["intensidad"].to_numpy(), pesos
    )

    ns_vals = df_sensores_limpio["nivel_servicio"].to_numpy()[np.newaxis, :]
    ns_en_radio = np.where(mascara_efectiva & ~np.isnan(ns_vals), ns_vals, -1.0)
    nivel_servicio_max = ns_en_radio.max(axis=1)
    nivel_servicio_max = np.where(
        nivel_servicio_max >= 0, nivel_servicio_max, np.nan
    )

    # Ingeniería de Características Avanzada (Modelo 2 / Tráfico Urbano)
    # 1. Índice de Fricción Vial reutilizando src.utils
    indice_friccion = calcular_indice_friccion(ocupacion_idw, carga_idw, intensidad_idw)

    # 2. Presión de Congestión Ponderada (Carga ponderada por cercanía directa)
    pesos_cercania = np.where(matriz_dist_m <= 250.0, 1.5, 1.0)
    pesos_presion = np.where(mascara_efectiva, pesos_cercania / np.maximum(matriz_dist_m, 10.0), 0.0)
    presion_idw = _media_ponderada_idw(df_sensores_limpio["carga"].to_numpy(), pesos_presion)

    # 3. Régimen de Tráfico Categórico reutilizando src.utils
    regimen = calcular_regimen_trafico(carga_idw, intensidad_idw)

    df_resultado = pd.DataFrame(
        {
            "stop_id": df_stops["stop_id"].astype(str).values,
            "timestamp_trafico": str(df_sensores_limpio["timestamp_trafico"].iloc[0]),
            "distancia_sensor_mas_cercano_m": np.round(dist_min_m, 1),
            "num_sensores_500m": num_sensores_500m.astype(int),
            "tiene_cobertura_trafico_500m": tiene_principal,
            "trafico_carga_entorno": np.round(carga_idw, 2),
            "trafico_ocupacion_entorno": np.round(ocupacion_idw, 2),
            "trafico_intensidad_entorno": np.round(intensidad_idw, 1),
            "trafico_nivel_servicio_max": nivel_servicio_max,
            "trafico_regimen": regimen,
            "trafico_indice_friccion": np.round(indice_friccion, 3),
            "trafico_presion_entorno": np.round(presion_idw, 2),
        }
    )
    return df_resultado
