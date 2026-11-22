"""
Módulo de Ingesta Meteorológica en Tiempo Real (Open-Meteo API - Madrid).
Captura cada 15 minutos las condiciones meteorológicas de superficie en Madrid
(temperatura, precipitación, lluvia, nieve, visibilidad, viento, rachas, humedad,
código WMO e indicador día/noche) sin requerir clave API.
"""

import logging
from datetime import datetime, timezone

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

URL_OPEN_METEO = "".join(["https://", "api.open-meteo.com", "/v1/forecast"])
MADRID_LAT = 40.4168
MADRID_LON = -3.7038


def obtener_meteorologia_actual(
    lat: float = MADRID_LAT,
    lon: float = MADRID_LON,
    timeout: int = 20
) -> pd.DataFrame:
    """
    Descarga las condiciones meteorológicas actuales en Madrid desde Open-Meteo
    (actualización cada 15 minutos) y retorna un DataFrame de 1 fila con sello temporal.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "is_day",
            "precipitation",
            "rain",
            "snowfall",
            "weather_code",
            "visibility",
            "wind_speed_10m",
            "wind_gusts_10m"
        ]),
        "timezone": "Europe/Madrid"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProyectoPuntualidadEMT/1.0"
    }

    resp = requests.get(URL_OPEN_METEO, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    datos = resp.json()

    actual = datos.get("current", {})
    ahora_utc = datetime.now(timezone.utc)

    df_meteo = pd.DataFrame([{
        "timestamp_captura_utc": ahora_utc.isoformat(),
        "timestamp_meteo_local": actual.get("time"),
        "temperatura_c": actual.get("temperature_2m"),
        "sensacion_termica_c": actual.get("apparent_temperature"),
        "humedad_relativa_pct": actual.get("relative_humidity_2m"),
        "es_de_dia": actual.get("is_day"),
        "precipitacion_mm": actual.get("precipitation"),
        "lluvia_mm": actual.get("rain"),
        "nevadas_cm": actual.get("snowfall"),
        "visibilidad_m": actual.get("visibility"),
        "codigo_tiempo_wmo": actual.get("weather_code"),
        "velocidad_viento_kmh": actual.get("wind_speed_10m"),
        "rachas_viento_kmh": actual.get("wind_gusts_10m")
    }])

    logging.info(
        f"Meteorología Madrid ({actual.get('time')}): "
        f"{actual.get('temperature_2m')}°C | Lluvia: {actual.get('rain')} mm | "
        f"Visibilidad: {actual.get('visibility')} m | Viento: {actual.get('wind_speed_10m')} km/h"
    )
    return df_meteo
