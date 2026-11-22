"""
Módulo de Funciones Auxiliares y Utilidades Matemáticas del Proyecto.
Centraliza algoritmos genéricos reutilizables en múltiples fases del pipeline.
"""

import base64
import gzip
import hashlib
import io
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi

from src.config import (
    HF_DATASET_ID_DEFECTO,
    TABLAS_GTFS_OPERATIVAS,
    URL_GTFS_OFICIAL_EMT,
)


def punto_en_poligono(x, y, poligono):
    """
    Algoritmo matemático nativo de Ray-Casting para determinar si un punto (x, y)
    se encuentra dentro de un polígono irregular de N vértices.
    Evita la importación de librerías gráficas pesadas en entornos de producción.
    """
    dentro = False
    n = len(poligono)
    if n == 0:
        return dentro
    p1x, p1y = poligono[0]
    for i in range(n + 1):
        p2x, p2y = poligono[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        dentro = not dentro
        p1x, p1y = p2x, p2y
    return dentro

def clasificar_regimen_horario(codigo_corto):
    """
    Segmenta las líneas de la EMT de Madrid en tres regímenes de operación
    basándose en las convenciones nomencláteras del estándar comercial GTFS.
    """
    codigo_str = str(codigo_corto).strip().upper()
    if codigo_str.startswith('N'):
        return 'Servicio Nocturno (Búho)'
    elif codigo_str == '203' or 'AEROPUERTO' in codigo_str:
        return 'Servicio 24 Horas'
    else:
        return 'Servicio Diurno Regular'

def extraer_calle_principal(nombre_parada):
    """
    Aísla la arteria vial principal del literal comercial de la parada de la EMT
    mediante expresiones regulares (Regex), eliminando cruces secundarios y portales.
    """
    txt = str(nombre_parada).strip()
    partes = re.split(r'\s*-\s*|\s+AND\s+|\s+Y\s+', txt, maxsplit=1)
    calle = partes[0].strip()
    calle = re.sub(r'\s*\d+$', '', calle).strip()
    return calle


def categorizar_tramos(total_marquesinas):
    """
    Segmenta las arterias viales de Madrid en tramos discretos según
    la densidad y frecuencia de marquesinas que soportan.
    """
    if total_marquesinas <= 3:
        return 'Capilaridad Local\n(1-3 paradas)'
    elif total_marquesinas <= 10:
        return 'Ejes Secundarios\n(4-10 paradas)'
    else:
        return 'Grandes Arterias\n(>10 paradas)'

def calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en línea recta en metros entre dos puntos geográficos
    utilizando la fórmula del Haversine, optimizada para matrices de NumPy.
    """
    R = 6371000.0  # Radio medio de la Tierra en metros
    rad_lat1, rad_lon1 = np.radians(lat1), np.radians(lon1)
    rad_lat2, rad_lon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = rad_lat2 - rad_lat1, rad_lon2 - rad_lon1
    a = np.sin(dlat/2.0)**2 + np.cos(rad_lat1) * np.cos(rad_lat2) * np.sin(dlon/2.0)**2
    return R * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))

def clasificar_extension(km):
    """
    Clasifica los trayectos comerciales de la EMT según su extensión kilométrica total.
    """
    if km <= 5.0:
        return 'Líneas Cortas / Barrio\n(0-5 km)'
    elif km <= 15.0:
        return 'Líneas Medias / Regulares\n(5-15 km)'
    else:
        return 'Largo Recorrido\n(>15 km)'

def clasificar_territorio_avanzado(ratio):
    """
    Determina el nivel de complejidad vial y fricción urbana de un trayecto
    combinando su índice de tortuosidad y la densidad de su muestreo geométrico.
    """
    if ratio == 1.0:
        return 'Interior Puro\n(Centro)'
    elif ratio == 0.0:
        return 'Exterior Puro\n(Periferia)'
    elif ratio <= 0.30:
        return 'Transversal\nPeriférica (0-30%)'
    elif ratio <= 0.60:
        return 'Transversal\nEquilibrada (30-60%)'
    else:
        return 'Transversal\nInvasiva (>60%)'

def firmar_grafico_emt(fig) -> None:
    """Añade la atribución oficial exigida por MobilityLabs EMT al pie de la figura."""
    fig.text(
        0.99,
        0.005,
        "Fuente: Powered by EMT de Madrid (http://emtmadrid.es) | Ayuntamiento de Madrid | AEMET",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
        style="italic",
    )

def anexar_parquet_deduplicado(df_nuevo: pd.DataFrame, ruta_parquet: Path, cols_dedup: list[str]) -> int:
    """Anexa filas nuevas a un fichero Parquet evitando duplicados y retorna filas finales."""
    ruta_parquet.parent.mkdir(parents=True, exist_ok=True)
    if ruta_parquet.exists():
        df_existente = pd.read_parquet(ruta_parquet)
        df_total = pd.concat([df_existente, df_nuevo], ignore_index=True)
    else:
        df_total = df_nuevo.copy()

    df_total = df_total.drop_duplicates(subset=cols_dedup, keep="last").reset_index(drop=True)
    df_total.to_parquet(ruta_parquet, index=False)
    return len(df_total)


def parsear_xml_trafico_gzip(xml_gzip_b64: str, ts_captura_utc: str) -> pd.DataFrame:
    """Descomprime un boletín XML GZIP de D1 y devuelve los sensores operativos saneados."""
    comprimido = base64.b64decode(xml_gzip_b64)
    xml_texto = gzip.decompress(comprimido).decode("utf-8", errors="replace")
    xml_limpio = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", xml_texto)
    xml_limpio = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", xml_limpio)
    raiz = ET.fromstring(xml_limpio)

    fecha_hora_el = raiz.find("fecha_hora")
    fecha_boletin = fecha_hora_el.text.strip() if fecha_hora_el is not None and fecha_hora_el.text else ""

    registros = list()
    for pm in raiz.findall("pm"):
        error_sensor = (pm.findtext("error") or "").strip()
        if error_sensor != "N":
            continue
        registros.append({
            "timestamp_captura_utc": ts_captura_utc,
            "fecha_hora_boletin": fecha_boletin,
            "idelem": (pm.findtext("idelem") or "").strip(),
            "descripcion": (pm.findtext("descripcion") or "").strip(),
            "tipo_elem": (pm.findtext("tipo_elem") or "").strip(),
            "intensidad": pd.to_numeric(pm.findtext("intensidad"), errors="coerce"),
            "ocupacion": pd.to_numeric(pm.findtext("ocupacion"), errors="coerce"),
            "carga": pd.to_numeric(pm.findtext("carga"), errors="coerce"),
            "nivelServicio": pd.to_numeric(pm.findtext("nivelServicio"), errors="coerce"),
            "velocidad": pd.to_numeric(pm.findtext("velocidad"), errors="coerce"),
            "st_x": pd.to_numeric((pm.findtext("st_x") or "").replace(",", "."), errors="coerce"),
            "st_y": pd.to_numeric((pm.findtext("st_y") or "").replace(",", "."), errors="coerce"),
        })
    return pd.DataFrame(registros)

def descargar_zip_gtfs_oficial(timeout: int = 90) -> bytes:
    """Descarga el paquete ZIP oficial gtfs_emt.zip desde el portal CKAN de EMT Madrid."""
    resp_zip = requests.get(
        URL_GTFS_OFICIAL_EMT,
        headers={"User-Agent": "Mozilla/5.0 ProyectoPuntualidadEMT/1.0"},
        timeout=timeout,
    )
    resp_zip.raise_for_status()
    if not zipfile.is_zipfile(io.BytesIO(resp_zip.content)):
        raise ValueError("El archivo descargado desde datos.emtmadrid.es no es un ZIP válido.")
    return resp_zip.content


def calcular_hashes_operativos_zip(zip_bytes: bytes) -> dict[str, str]:
    """Calcula el SHA-256 de las 8 tablas operativas del paquete GTFS (excluye agency y feed_info)."""
    hashes = dict()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for nombre in sorted(zf.namelist()):
            nombre_limpio = Path(nombre).name
            if nombre_limpio in TABLAS_GTFS_OPERATIVAS:
                hashes[nombre_limpio] = hashlib.sha256(zf.read(nombre)).hexdigest()
    return hashes


def obtener_cliente_hf() -> tuple[HfApi, str]:
    """Inicializa el cliente oficial de Hugging Face y valida credenciales desde variables de entorno."""
    hf_token = os.getenv("HF_TOKEN", "").strip()
    dataset_id = os.getenv("HF_DATASET_ID", HF_DATASET_ID_DEFECTO).strip() or HF_DATASET_ID_DEFECTO
    if not hf_token:
        raise ValueError("Falta HF_TOKEN en las variables de entorno o archivo .env")
    return HfApi(token=hf_token), dataset_id
