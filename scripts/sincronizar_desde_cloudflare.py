"""
Sincronizador Seguro Nube -> Local (Cloudflare D1 -> data/raw/realtime/*.parquet).
1. Descarga lotes pendientes de EMT, Tráfico (GZIP XML) y Meteorología desde Cloudflare.
2. Descomprime el XML de tráfico y filtra únicamente los sensores operativos.
3. Guarda/anexa los registros en los ficheros diarios .parquet locales.
4. Verifica la escritura en disco y ordena a Cloudflare purgar solo los IDs descargados.
"""

import base64
import gzip
import logging
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DIR_REALTIME = Path("data/raw/realtime")
DIR_EMT = DIR_REALTIME / "emt"
DIR_TRAFICO = DIR_REALTIME / "trafico"
DIR_METEO = DIR_REALTIME / "meteorologia"


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


def ejecutar_sincronizacion() -> None:
    """Punto de entrada principal de sincronización nube -> disco local."""
    base_url = os.getenv("CLOUDFLARE_WORKER_URL", "").strip().rstrip("/")
    token = os.getenv("SYNC_SECRET_TOKEN", "").strip()
    if not base_url or not token:
        raise ValueError("Faltan CLOUDFLARE_WORKER_URL o SYNC_SECRET_TOKEN en el archivo .env")

    logging.info("Iniciando sincronización segura desde Cloudflare D1...")
    for tabla in ("emt_llegadas", "trafico_xml_raw", "meteo_madrid"):
        n = sincronizar_tabla(base_url, token, tabla)
        logging.info(f"Resumen [{tabla}]: {n} registros nuevos incorporados a data/raw/realtime/.")


if __name__ == "__main__":
    ejecutar_sincronizacion()
