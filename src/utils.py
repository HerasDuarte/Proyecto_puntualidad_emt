"""
Módulo de Funciones Auxiliares y Utilidades Matemáticas del Proyecto.
Centraliza algoritmos genéricos reutilizables en múltiples fases del pipeline.
"""

import base64
import csv
import gzip
import hashlib
import io
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi

from src.config import (
    HF_DATASET_ID_DEFECTO,
    TABLAS_GTFS_OPERATIVAS,
    URL_GTFS_OFICIAL_EMT,
)

ZONA_MADRID = ZoneInfo("Europe/Madrid")
HEADERS_HTTP_PROYECTO = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "ProyectoPuntualidadEMT/1.0"
    ),
    "Accept": "application/json",
}


def formatear_hora_madrid(ts_iso: str | None) -> str:
    """Convierte un timestamp ISO UTC a hora local de Madrid legible."""
    if not ts_iso:
        return "Sin datos"
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.astimezone(ZONA_MADRID).strftime("%d/%m/%Y %H:%M:%S")
    except ValueError:
        return ts_iso


def extraer_contexto_madrid(ts_iso: str | None = None) -> tuple[str, int]:
    """Devuelve (tipo_dia GTFS, hora 0-23) en horario local de Madrid."""
    try:
        dt = (
            datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
            if ts_iso
            else datetime.now(ZONA_MADRID)
        ).astimezone(ZONA_MADRID)
    except ValueError:
        dt = datetime.now(ZONA_MADRID)
    dia_sem = dt.weekday()
    tipo_dia = "SA" if dia_sem == 5 else ("FE" if dia_sem == 6 else "LA")
    return tipo_dia, dt.hour


def sanear_xml_madrid(contenido_bytes: bytes) -> str:
    """Decodifica el XML municipal y sanea caracteres de control o '&' sueltos."""
    try:
        texto = contenido_bytes.decode("utf-8")
    except UnicodeDecodeError:
        texto = contenido_bytes.decode("latin-1", errors="replace")

    texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", texto)
    texto = re.sub(
        r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)",
        "&amp;",
        texto,
    )
    return texto


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
    if not xml_gzip_b64:
        return pd.DataFrame()
    try:
        comprimido = base64.b64decode(xml_gzip_b64)
        xml_limpio = sanear_xml_madrid(gzip.decompress(comprimido))
        raiz = ET.fromstring(xml_limpio)
    except (ValueError, OSError, ET.ParseError) as exc:
        logging.warning("Boletín XML de tráfico inválido o corrupto: %s", exc)
        return pd.DataFrame()

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
    df_res = pd.DataFrame(registros)
    if df_res.empty:
        return df_res
    for col in ("intensidad", "ocupacion", "carga", "nivelServicio", "velocidad"):
        df_res.loc[df_res[col] < 0, col] = np.nan
    for col_pct in ("ocupacion", "carga"):
        df_res.loc[df_res[col_pct] > 100, col_pct] = np.nan
    return df_res

def descargar_zip_gtfs_oficial(
    timeout: int = 90, reintentos: int = 3
) -> bytes:
    """Descarga gtfs_emt.zip desde CKAN EMT Madrid con reintentos ante fallos de red."""
    ultimo_error: Exception | None = None
    for intento in range(1, max(reintentos, 1) + 1):
        try:
            resp_zip = requests.get(
                URL_GTFS_OFICIAL_EMT,
                headers={"User-Agent": HEADERS_HTTP_PROYECTO["User-Agent"]},
                timeout=timeout,
            )
            resp_zip.raise_for_status()
            if not zipfile.is_zipfile(io.BytesIO(resp_zip.content)):
                raise ValueError(
                    "El archivo descargado desde datos.emtmadrid.es no es un ZIP válido."
                )
            return resp_zip.content
        except (requests.RequestException, ValueError) as exc:
            ultimo_error = exc
            if intento < reintentos:
                espera = 2 * intento
                logging.warning(
                    "Reintento %d/%d descargando GTFS en %ds (%s).",
                    intento,
                    reintentos,
                    espera,
                    exc,
                )
                time.sleep(espera)
    raise RuntimeError(
        f"No se pudo descargar gtfs_emt.zip tras {reintentos} intentos: {ultimo_error}"
    )


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


def _leer_csv_desde_zip(zf: zipfile.ZipFile, nombre_archivo: str) -> list[dict[str, str]]:
    """Lee un archivo .txt dentro de un ZIP GTFS buscando coincidencia por nombre."""
    objetivo = next(
        (n for n in zf.namelist() if Path(n).name == nombre_archivo),
        None,
    )
    if not objetivo:
        return []
    with zf.open(objetivo, "r") as fh:
        lector = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8-sig"))
        return list(lector)


def _hora_gtfs_a_segundos(valor: str | None) -> int | None:
    """Convierte una hora GTFS HH:MM:SS a segundos o None si es inválida."""
    if not valor:
        return None
    partes = valor.strip().split(":")
    if len(partes) != 3:
        return None
    try:
        return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
    except ValueError:
        return None


def comparar_zips_gtfs(
    zip_anterior_bytes: bytes | None, zip_nuevo_bytes: bytes
) -> dict[str, object]:
    """Compara dos paquetes ZIP GTFS y resume altas, bajas y variaciones clave."""
    def _extraer_metricas(raw: bytes) -> dict[str, object]:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            stops = {
                (r.get("stop_id") or "").strip()
                for r in _leer_csv_desde_zip(zf, "stops.txt")
                if (r.get("stop_id") or "").strip()
            }
            routes = {
                (r.get("route_short_name") or r.get("route_id") or "").strip()
                for r in _leer_csv_desde_zip(zf, "routes.txt")
                if (r.get("route_short_name") or r.get("route_id") or "").strip()
            }
            trips = _leer_csv_desde_zip(zf, "trips.txt")
            freqs = _leer_csv_desde_zip(zf, "frequencies.txt")
            return {
                "stops": stops,
                "routes": routes,
                "n_trips": len(trips),
                "n_freqs": len(freqs),
            }

    nuevo = _extraer_metricas(zip_nuevo_bytes)
    if not zip_anterior_bytes:
        return {
            "total_paradas": len(nuevo["stops"]),
            "total_lineas": len(nuevo["routes"]),
            "total_viajes": nuevo["n_trips"],
            "paradas_altas": 0,
            "paradas_bajas": 0,
            "lineas_altas": 0,
            "lineas_bajas": 0,
            "viajes_delta": 0,
            "frecuencias_delta": 0,
        }

    prev = _extraer_metricas(zip_anterior_bytes)
    altas_p = sorted(nuevo["stops"] - prev["stops"])
    bajas_p = sorted(prev["stops"] - nuevo["stops"])
    altas_l = sorted(nuevo["routes"] - prev["routes"])
    bajas_l = sorted(prev["routes"] - nuevo["routes"])

    return {
        "total_paradas": len(nuevo["stops"]),
        "total_lineas": len(nuevo["routes"]),
        "total_viajes": nuevo["n_trips"],
        "paradas_altas": len(altas_p),
        "paradas_bajas": len(bajas_p),
        "lineas_altas": len(altas_l),
        "lineas_bajas": len(bajas_l),
        "viajes_delta": int(nuevo["n_trips"]) - int(prev["n_trips"]),
        "frecuencias_delta": int(nuevo["n_freqs"]) - int(prev["n_freqs"]),
        "ejemplos_paradas_altas": altas_p[:8],
        "ejemplos_paradas_bajas": bajas_p[:8],
        "ejemplos_lineas_altas": altas_l[:8],
        "ejemplos_lineas_bajas": bajas_l[:8],
    }


def auditar_cuarentenas_gtfs_zip(
    zip_bytes: bytes,
) -> dict[str, dict[str, object]]:
    """Audita integridad referencial y física del ZIP GTFS devolviendo cuarentenas y motivo."""
    incidencias: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        routes_rows = _leer_csv_desde_zip(zf, "routes.txt")
        stops_rows = _leer_csv_desde_zip(zf, "stops.txt")
        cal_rows = _leer_csv_desde_zip(zf, "calendar.txt")
        cal_dates_rows = _leer_csv_desde_zip(zf, "calendar_dates.txt")
        trips_rows = _leer_csv_desde_zip(zf, "trips.txt")
        freq_rows = _leer_csv_desde_zip(zf, "frequencies.txt")

    route_ids = {
        (r.get("route_id") or "").strip()
        for r in routes_rows
        if (r.get("route_id") or "").strip()
    }
    service_ids = {
        (r.get("service_id") or "").strip()
        for r in cal_rows + cal_dates_rows
        if (r.get("service_id") or "").strip()
    }
    trip_ids_validos: set[str] = set()

    stops_invalidas = 0
    for r in stops_rows:
        try:
            lat = float(r.get("stop_lat", ""))
            lon = float(r.get("stop_lon", ""))
            if not (40.1 <= lat <= 40.8 and -4.2 <= lon <= -3.2):
                stops_invalidas += 1
        except ValueError:
            stops_invalidas += 1
    if stops_invalidas > 0:
        incidencias["stops.txt"] = {
            "cantidad": stops_invalidas,
            "motivo": "Coordenadas vacías o fuera del ámbito geográfico de Madrid",
        }

    trips_huerfanos = 0
    for r in trips_rows:
        tid = (r.get("trip_id") or "").strip()
        rid = (r.get("route_id") or "").strip()
        sid = (r.get("service_id") or "").strip()
        if not tid or rid not in route_ids or sid not in service_ids:
            trips_huerfanos += 1
        else:
            trip_ids_validos.add(tid)
    if trips_huerfanos > 0:
        incidencias["trips.txt"] = {
            "cantidad": trips_huerfanos,
            "motivo": "Viajes huérfanos sin route_id o service_id válido",
        }

    freq_huerfanas = 0
    freq_invertidas = 0
    for r in freq_rows:
        tid = (r.get("trip_id") or "").strip()
        ini = _hora_gtfs_a_segundos(r.get("start_time"))
        fin = _hora_gtfs_a_segundos(r.get("end_time"))
        try:
            headway = float(r.get("headway_secs", "0"))
        except ValueError:
            headway = 0.0
        if tid not in trip_ids_validos:
            freq_huerfanas += 1
        elif ini is None or fin is None or ini >= fin or headway <= 0:
            freq_invertidas += 1

    if freq_huerfanas + freq_invertidas > 0:
        motivos = []
        if freq_huerfanas > 0:
            motivos.append(f"{freq_huerfanas} huérfanas sin trip_id válido")
        if freq_invertidas > 0:
            motivos.append(f"{freq_invertidas} con franja invertida o headway <= 0")
        incidencias["frequencies.txt"] = {
            "cantidad": freq_huerfanas + freq_invertidas,
            "motivo": "; ".join(motivos),
        }

    return incidencias


def calcular_indice_friccion(ocupacion: np.ndarray | float, carga: np.ndarray | float, intensidad: np.ndarray | float) -> np.ndarray | float:
    """Calcula el índice de fricción vial penalizando la alta ocupación/carga con baja intensidad."""
    intensidad_segura = np.maximum(intensidad, 1.0)
    return (ocupacion * carga) / intensidad_segura


def calcular_regimen_trafico(carga: np.ndarray | float, intensidad: np.ndarray | float) -> np.ndarray | str:
    """Segmenta el régimen de tráfico en función de la saturación (carga) y el flujo (intensidad)."""
    return np.select(
        [
            (carga >= 70.0) & (intensidad < 200.0),
            (carga >= 40.0) & (intensidad >= 200.0),
            (carga < 40.0) & (intensidad >= 200.0),
        ],
        [
            "Colapso_Estatico",
            "Saturado_Con_Movimiento",
            "Fluido",
        ],
        default="Libre",
    )
