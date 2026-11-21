"""
Sincronizador Seguro Nube -> Local + Vigilancia y Reconstrucción Automática Diaria GTFS.
1. Descarga lotes pendientes de EMT, Tráfico (GZIP XML) y Meteorología desde Cloudflare D1.
2. Descomprime el XML de tráfico y filtra únicamente los sensores operativos.
3. Guarda/anexa los registros en data/raw/realtime/*.parquet y purga en Cloudflare D1.
4. Audita diariamente las 8 tablas operativas de los EDAs GTFS (SHA-256),
   excluyendo agency.txt y feed_info.txt, y relanza main.py para reconstruir
   maestro_gtfs_YYYYMMDD.parquet si cambia cualquier tabla operativa.
"""

import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DIR_RAW = Path("data/raw")
DIR_PROCESSED = Path("data/processed")
DIR_REALTIME = DIR_RAW / "realtime"
DIR_EMT = DIR_REALTIME / "emt"
DIR_TRAFICO = DIR_REALTIME / "trafico"
DIR_METEO = DIR_REALTIME / "meteorologia"

RUTA_HISTORIAL_GTFS = DIR_PROCESSED / "historial_versiones_gtfs.json"

# Las 8 tablas exactas auditadas en los EDAs 01-08 (se excluyen agency.txt y feed_info.txt)
TABLAS_GTFS_OPERATIVAS = (
    "routes.txt",
    "stops.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "trips.txt",
    "frequencies.txt",
    "shapes.txt",
    "stop_times.txt",
)


def _anexar_parquet(df_nuevo: pd.DataFrame, ruta_parquet: Path, cols_dedup: list[str]) -> int:
    """Anexa filas nuevas a un Parquet diario evitando duplicados y retorna filas finales."""
    ruta_parquet.parent.mkdir(parents=True, exist_ok=True)
    if ruta_parquet.exists():
        df_existente = pd.read_parquet(ruta_parquet)
        df_total = pd.concat([df_existente, df_nuevo], ignore_index=True)
    else:
        df_total = df_nuevo.copy()

    df_total = df_total.drop_duplicates(subset=cols_dedup, keep="last").reset_index(drop=True)
    df_total.to_parquet(ruta_parquet, index=False)
    return len(df_total)


def _parsear_xml_trafico_gzip(xml_gzip_b64: str, ts_captura_utc: str) -> pd.DataFrame:
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


def sincronizar_tabla(base_url: str, token: str, tabla: str) -> int:
    """Descarga todos los lotes pendientes de una tabla D1, guarda en Parquet y purga en nube."""
    total_descargadas = 0
    while True:
        resp = requests.get(
            f"{base_url}/api/export",
            params={"token": token, "table": tabla},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        filas = payload.get("rows", list())
        if not filas:
            break

        max_id = max(int(r["id"]) for r in filas)
        df = pd.DataFrame(filas)

        if tabla == "emt_llegadas":
            df["fecha_dia"] = df["timestamp_captura_utc"].str.slice(0, 10).str.replace("-", "")
            for dia, grupo in df.groupby("fecha_dia"):
                ruta = DIR_EMT / f"emt_llegadas_{dia}.parquet"
                cols = [c for c in grupo.columns if c not in ("id", "fecha_dia")]
                _anexar_parquet(grupo[cols], ruta, ["timestamp_captura_utc", "stop_id", "line", "bus_id"])

        elif tabla == "meteo_madrid":
            df["fecha_dia"] = df["timestamp_captura_utc"].str.slice(0, 10).str.replace("-", "")
            for dia, grupo in df.groupby("fecha_dia"):
                ruta = DIR_METEO / f"meteo_madrid_{dia}.parquet"
                cols = [c for c in grupo.columns if c not in ("id", "fecha_dia")]
                _anexar_parquet(grupo[cols], ruta, ["timestamp_captura_utc"])

        elif tabla == "trafico_xml_raw":
            for _, fila in df.iterrows():
                ts_utc = str(fila["timestamp_captura_utc"])
                dia = ts_utc[:10].replace("-", "")
                df_sensores = _parsear_xml_trafico_gzip(str(fila["xml_gzip_base64"]), ts_utc)
                if not df_sensores.empty:
                    ruta = DIR_TRAFICO / f"trafico_sensores_{dia}.parquet"
                    _anexar_parquet(df_sensores, ruta, ["timestamp_captura_utc", "idelem"])

        resp_purge = requests.post(
            f"{base_url}/api/purge",
            params={"token": token},
            json={"table": tabla, "max_id": max_id},
            timeout=30,
        )
        resp_purge.raise_for_status()
        total_descargadas += len(filas)
        logging.info(f"[{tabla}] Sincronizadas y liberadas {len(filas)} filas (hasta ID {max_id}).")

    return total_descargadas

def _descargar_zip_gtfs_oficial() -> bytes:
    """Descarga directamente el paquete ZIP oficial gtfs_emt.zip desde datos.emtmadrid.es."""
    url_gtfs_oficial = (
        "https://" + "datos." + "emtmadrid.es/dataset/9b23259a-4491-494b-9695-36a7709b2c12"
        "/resource/3cba2058-9833-422c-a704-bf992d31d2ee/download/gtfs_emt.zip"
    )
    resp_zip = requests.get(
        url_gtfs_oficial,
        headers={"User-Agent": "Mozilla/5.0 ProyectoPuntualidadEMT/1.0"},
        timeout=90,
    )
    resp_zip.raise_for_status()

    if not zipfile.is_zipfile(io.BytesIO(resp_zip.content)):
        raise ValueError("El archivo descargado desde datos.emtmadrid.es no es un ZIP válido.")
    return resp_zip.content

def _calcular_hashes_operativos_zip(zip_bytes: bytes) -> dict[str, str]:
    """Calcula el SHA-256 de las 8 tablas operativas de los EDAs GTFS."""
    hashes = dict()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for nombre in sorted(zf.namelist()):
            nombre_limpio = Path(nombre).name
            if nombre_limpio in TABLAS_GTFS_OPERATIVAS:
                hashes[nombre_limpio] = hashlib.sha256(zf.read(nombre)).hexdigest()
    return hashes

def _leer_claves_cuarentena() -> dict[str, set[str]]:
    """Extrae los trip_id y route_id presentes en los archivos quarantine_*.csv actuales."""
    claves = {"trip_id": set(), "route_id": set()}
    for archivo_q in DIR_PROCESSED.glob("quarantine_*.csv"):
        try:
            df_q = pd.read_csv(archivo_q, dtype=str)
            if "trip_id" in df_q.columns:
                claves["trip_id"].update(df_q["trip_id"].dropna().unique())
            if "route_id" in df_q.columns:
                claves["route_id"].update(df_q["route_id"].dropna().unique())
        except Exception as exc:
            logging.warning(f"[GTFS] No se pudo leer {archivo_q.name} durante reconciliación: {exc}")
    return claves


def reconciliar_cuarentenas_gtfs(claves_previas: dict[str, set[str]], ruta_maestro: Path) -> int:
    """
    Verifica si los registros que estaban en cuarentena ya tienen correspondencia completa
    en el nuevo maestro_gtfs.parquet. Purga los archivos quarantine_*.csv obsoletos una vez
    confirmado su rescate y devuelve el total de claves recuperadas.
    """
    if not claves_previas["trip_id"] and not claves_previas["route_id"]:
        for archivo_q in DIR_PROCESSED.glob("quarantine_*.csv"):
            df_q = pd.read_csv(archivo_q)
            if df_q.empty:
                archivo_q.unlink(missing_ok=True)
        return 0

    df_maestro_claves = pd.read_parquet(ruta_maestro, columns=["trip_id", "route_id"])
    trips_maestro = set(df_maestro_claves["trip_id"].astype(str).unique())
    routes_maestro = set(df_maestro_claves["route_id"].astype(str).unique())

    trips_rescatados = claves_previas["trip_id"].intersection(trips_maestro)
    routes_rescatadas = claves_previas["route_id"].intersection(routes_maestro)
    total_rescatados = len(trips_rescatados) + len(routes_rescatadas)

    if total_rescatados > 0:
        logging.info(
            f"[GTFS] Reconciliación de cuarentena exitosa: {len(trips_rescatados)} trip_id y "
            f"{len(routes_rescatadas)} route_id recuperados e integrados en {ruta_maestro.name}."
        )

    for archivo_q in list(DIR_PROCESSED.glob("quarantine_*.csv")):
        df_q = pd.read_csv(archivo_q, dtype=str)
        if df_q.empty:
            archivo_q.unlink(missing_ok=True)
            continue

        if "trip_id" in df_q.columns:
            df_q = df_q[~df_q["trip_id"].isin(trips_maestro)]
        if "route_id" in df_q.columns:
            df_q = df_q[~df_q["route_id"].isin(routes_maestro)]

        if df_q.empty:
            archivo_q.unlink(missing_ok=True)
            logging.info(f"[GTFS] Archivo de cuarentena obsoleto eliminado tras rescate completo: {archivo_q.name}")
        else:
            df_q.to_csv(archivo_q, index=False)

    return total_rescatados

def auditar_y_actualizar_gtfs_diario() -> bool:
    """
    Audita diariamente las 8 tablas operativas del GTFS oficial de EMT Madrid.
    Si detecta cambios en alguna tabla operativa, guarda copia versionada,
    ejecuta scripts/pipeline_limpieza.py + scripts/construir_maestro_gtfs.py
    tras respaldar el maestro_gtfs.parquet anterior, y reconcilia los registros
    en cuarentena contra la nueva tabla maestra.
    """
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    historial = list()
    if RUTA_HISTORIAL_GTFS.exists():
        historial = json.loads(RUTA_HISTORIAL_GTFS.read_text(encoding="utf-8"))

    try:
        zip_bytes = _descargar_zip_gtfs_oficial()
    except Exception as exc:
        logging.warning(f"[GTFS] No se pudo verificar actualización diaria hoy ({exc}). Se mantiene versión vigente.")
        return False

    hashes_remotos = _calcular_hashes_operativos_zip(zip_bytes)

    if not historial:
        hashes_locales = dict()
        dir_gtfs_origen = DIR_RAW / "gtfs"
        for tabla in TABLAS_GTFS_OPERATIVAS:
            ruta_tabla = dir_gtfs_origen / tabla
            if ruta_tabla.exists():
                hashes_locales[tabla] = hashlib.sha256(ruta_tabla.read_bytes()).hexdigest()

        hashes_base = hashes_locales if hashes_locales else hashes_remotos
        historial.append({
            "version_tag": "20261118_inicial",
            "detectado_utc": "2026-11-18T00:00:00Z",
            "tablas_modificadas": list(hashes_base.keys()),
            "hashes_tablas": hashes_base,
            "ruta_maestro": str(DIR_PROCESSED / "maestro_gtfs.parquet"),
        })
        RUTA_HISTORIAL_GTFS.write_text(json.dumps(historial, indent=2, ensure_ascii=False), encoding="utf-8")

    ultimo_registro = historial[-1]
    hashes_previos = ultimo_registro.get("hashes_tablas", dict())

    tablas_cambiadas = [
        tabla for tabla, h in hashes_remotos.items()
        if hashes_previos.get(tabla) != h
    ]

    if not tablas_cambiadas:
        logging.info(f"[GTFS] Auditoría diaria OK: sin cambios en las 8 tablas operativas (vigente: {ultimo_registro['version_tag']}).")
        return False

    etiqueta_hoy = datetime.now(timezone.utc).strftime("%Y%m%d")
    dir_version = DIR_RAW / f"gtfs_{etiqueta_hoy}"
    dir_version.mkdir(parents=True, exist_ok=True)
    dir_gtfs_activo = DIR_RAW / "gtfs"
    dir_gtfs_activo.mkdir(parents=True, exist_ok=True)

    logging.warning(f"[GTFS] ¡Cambio detectado en tablas: {', '.join(tablas_cambiadas)}! Reconstruyendo versión {etiqueta_hoy}...")

    ruta_maestro_vigente = DIR_PROCESSED / "maestro_gtfs.parquet"
    if ruta_maestro_vigente.exists():
        ruta_respaldo = DIR_PROCESSED / f"maestro_gtfs_previo_{etiqueta_hoy}.parquet"
        if not ruta_respaldo.exists():
            ruta_respaldo.write_bytes(ruta_maestro_vigente.read_bytes())
            logging.info(f"[GTFS] Copia de seguridad creada en {ruta_respaldo}.")

    claves_cuarentena_previas = _leer_claves_cuarentena()

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for nombre in zf.namelist():
            nombre_limpio = Path(nombre).name
            if nombre_limpio in TABLAS_GTFS_OPERATIVAS:
                contenido = zf.read(nombre)
                (dir_version / nombre_limpio).write_bytes(contenido)
                (dir_gtfs_activo / nombre_limpio).write_bytes(contenido)

    subprocess.run([sys.executable, "scripts/pipeline_limpieza.py"], check=True)
    subprocess.run([sys.executable, "scripts/construir_maestro_gtfs.py"], check=True)

    reconciliar_cuarentenas_gtfs(claves_cuarentena_previas, ruta_maestro_vigente)

    ruta_maestro_versionado = DIR_PROCESSED / f"maestro_gtfs_{etiqueta_hoy}.parquet"
    ruta_maestro_versionado.write_bytes(ruta_maestro_vigente.read_bytes())

    historial.append({
        "version_tag": etiqueta_hoy,
        "detectado_utc": datetime.now(timezone.utc).isoformat(),
        "tablas_modificadas": tablas_cambiadas,
        "hashes_tablas": hashes_remotos,
        "ruta_maestro": str(ruta_maestro_versionado),
    })
    RUTA_HISTORIAL_GTFS.write_text(json.dumps(historial, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"[GTFS] Nueva versión {etiqueta_hoy} integrada y reconstruida en {ruta_maestro_versionado}.")
    return True

def ejecutar_sincronizacion() -> None:
    """Punto de entrada principal de sincronización nube -> disco local + vigilancia diaria GTFS."""
    base_url = os.getenv("CLOUDFLARE_WORKER_URL", "").strip().rstrip("/")
    token = os.getenv("SYNC_SECRET_TOKEN", "").strip()
    if not base_url or not token:
        raise ValueError("Faltan CLOUDFLARE_WORKER_URL o SYNC_SECRET_TOKEN en el archivo .env")

    logging.info("Iniciando sincronización segura desde Cloudflare D1...")
    for tabla in ("emt_llegadas", "trafico_xml_raw", "meteo_madrid"):
        n = sincronizar_tabla(base_url, token, tabla)
        logging.info(f"Resumen [{tabla}]: {n} registros nuevos incorporados a data/raw/realtime/.")

    auditar_y_actualizar_gtfs_diario()


if __name__ == "__main__":
    ejecutar_sincronizacion()