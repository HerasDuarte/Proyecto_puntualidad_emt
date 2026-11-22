"""
Script de ingesta en tiempo real del estado del tráfico del Ayuntamiento de Madrid.
Descarga el feed XML oficial (informo.madrid.es/informo/tmadrid/pm.xml), sanea posibles
entidades XML mal formadas, transforma las coordenadas UTM ETRS89 (EPSG:25830) a WGS84
(lat/lon) y guarda una instantánea estructurada en data/raw/sensores_trafico/sensores.csv.
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from pyproj import Transformer

from src.sensores_trafico import sanear_xml_madrid

URL_TRAFICO_XML = "https://" + "informo." + "madrid.es/informo/tmadrid/pm.xml"
OUTPUT_DIR = Path("data/raw/sensores_trafico")


def descargar_y_parsear_trafico():
    print("====== DESCARGANDO ESTADO DEL TRÁFICO EN VIVO (AYUNTAMIENTO DE MADRID) ======\n")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProyectoPuntualidadEMT/1.0"
    }
    resp = requests.get(URL_TRAFICO_XML, headers=headers, timeout=25)
    resp.raise_for_status()

    xml_limpio = sanear_xml_madrid(resp.content)
    raiz = ET.fromstring(xml_limpio)

    fecha_hora_dato = raiz.findtext("fecha_hora", default=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print(f"-> Timestamp oficial del feed de tráfico : {fecha_hora_dato}")

    # Transformador oficial de UTM ETRS89 Huso 30N (EPSG:25830) a GPS WGS84 (EPSG:4326)
    transformador = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)

    registros = []
    for pm in raiz.findall("pm"):
        idelem = pm.findtext("idelem")
        descripcion = pm.findtext("descripcion", default="").strip()
        acceso_asociado = pm.findtext("accesoAsociado", default="").strip()
        intensidad = pm.findtext("intensidad")
        ocupacion = pm.findtext("ocupacion")
        carga = pm.findtext("carga")
        nivel_servicio = pm.findtext("nivelServicio")
        intensidad_sat = pm.findtext("intensidadSat")
        error_sensor = pm.findtext("error", default="N").strip()
        subarea = pm.findtext("subarea", default="").strip()
        st_x_raw = pm.findtext("st_x", default="").replace(",", ".")
        st_y_raw = pm.findtext("st_y", default="").replace(",", ".")
        velocidad = pm.findtext("velocidad")

        lat, lon = None, None
        try:
            if st_x_raw and st_y_raw:
                x_val = float(st_x_raw)
                y_val = float(st_y_raw)
                lon, lat = transformador.transform(x_val, y_val)
        except ValueError:
            pass

        registros.append({
            "timestamp_feed": fecha_hora_dato,
            "idelem": idelem,
            "descripcion": descripcion,
            "acceso_asociado": acceso_asociado,
            "intensidad": pd.to_numeric(intensidad, errors="coerce"),
            "ocupacion": pd.to_numeric(ocupacion, errors="coerce"),
            "carga": pd.to_numeric(carga, errors="coerce"),
            "nivel_servicio": pd.to_numeric(nivel_servicio, errors="coerce"),
            "intensidad_sat": pd.to_numeric(intensidad_sat, errors="coerce"),
            "velocidad_trafico": pd.to_numeric(velocidad, errors="coerce"),
            "error_sensor": error_sensor,
            "subarea": subarea,
            "lat": lat,
            "lon": lon
        })

    df_trafico = pd.DataFrame(registros)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ruta_csv = OUTPUT_DIR / "sensores.csv"
    df_trafico.to_csv(ruta_csv, index=False, encoding="utf-8")

    print(f"-> Sensores capturados en toda la ciudad  : {len(df_trafico):,}")
    print(f"-> Sensores operativos sin error ('N')   : {(df_trafico['error_sensor'] == 'N').sum():,}")
    print(f"-> Archivo guardado en                   : {ruta_csv}\n")
    print("-> Primeras 5 filas capturadas:")
    print(df_trafico[["idelem", "descripcion", "intensidad", "ocupacion", "carga", "nivel_servicio", "lat", "lon"]].head())


if __name__ == "__main__":
    descargar_y_parsear_trafico()
