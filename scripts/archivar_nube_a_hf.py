"""
Archivador Cloud-to-Cloud: Cloudflare D1 -> Hugging Face Dataset Privado.
1. Sube a Hugging Face cualquier fichero .parquet local existente en data/raw/realtime/
   que aún no esté consolidado en el dataset remoto.
2. Descarga los lotes pendientes de Cloudflare D1 (emt_llegadas, trafico_xml_raw, meteo_madrid),
   los fusiona sin duplicados con los .parquet diarios del Dataset privado en Hugging Face,
   los sube y purga los registros procesados en Cloudflare D1.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.sincronizar_desde_cloudflare import _anexar_parquet, _parsear_xml_trafico_gzip

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATASET_ID_DEFECTO = "HerasDuarte/emt-madrid-realtime-data"
DIR_REALTIME_LOCAL = Path("data/raw/realtime")

CONFIG_TABLAS = {
    "emt_llegadas": {
        "subcarpeta": "emt",
        "prefijo": "emt_llegadas",
        "dedup": ["timestamp_captura_utc", "stop_id", "line", "bus_id"],
    },
    "trafico_xml_raw": {
        "subcarpeta": "trafico",
        "prefijo": "trafico_sensores",
        "dedup": ["timestamp_captura_utc", "idelem"],
    },
    "meteo_madrid": {
        "subcarpeta": "meteorologia",
        "prefijo": "meteo_madrid",
        "dedup": ["timestamp_captura_utc"],
    },
}


def _obtener_api_hf() -> tuple[HfApi, str]:
    """Inicializa el cliente oficial de Hugging Face y verifica credenciales."""
    hf_token = os.getenv("HF_TOKEN", "").strip()
    dataset_id = os.getenv("HF_DATASET_ID", DATASET_ID_DEFECTO).strip() or DATASET_ID_DEFECTO
    if not hf_token:
        raise ValueError("Falta HF_TOKEN en las variables de entorno o archivo .env")
    return HfApi(token=hf_token), dataset_id


def _descargar_parquet_remoto_si_existe(api: HfApi, dataset_id: str, ruta_remota: str, ruta_destino: Path) -> bool:
    """Descarga un .parquet diario desde Hugging Face si ya existe en el repositorio remoto."""
    try:
        descargado = hf_hub_download(
            repo_id=dataset_id,
            filename=ruta_remota,
            repo_type="dataset",
            token=api.token,
        )
        ruta_destino.parent.mkdir(parents=True, exist_ok=True)
        ruta_destino.write_bytes(Path(descargado).read_bytes())
        return True
    except EntryNotFoundError:
        return False


def sincronizar_locales_previos_a_hf(api: HfApi, dataset_id: str) -> int:
    """Sube los .parquet diarios que ya existan en data/raw/realtime/ hacia Hugging Face."""
    if not DIR_REALTIME_LOCAL.exists():
        return 0

    operaciones = list()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for cfg in CONFIG_TABLAS.values():
            sub = cfg["subcarpeta"]
            dedup = cfg["dedup"]
            carpeta_local = DIR_REALTIME_LOCAL / sub
            if not carpeta_local.exists():
                continue

            for fichero_local in sorted(carpeta_local.glob("*.parquet")):
                ruta_remota = f"realtime/{sub}/{fichero_local.name}"
                ruta_trabajo = tmp_path / sub / fichero_local.name
                df_local = pd.read_parquet(fichero_local)

                if _descargar_parquet_remoto_si_existe(api, dataset_id, ruta_remota, ruta_trabajo):
                    filas_antes = len(pd.read_parquet(ruta_trabajo))
                    filas_despues = _anexar_parquet(df_local, ruta_trabajo, dedup)
                    if filas_despues == filas_antes:
                        continue
                else:
                    ruta_trabajo.parent.mkdir(parents=True, exist_ok=True)
                    df_local.drop_duplicates(subset=dedup, keep="last").to_parquet(ruta_trabajo, index=False)

                operaciones.append(
                    CommitOperationAdd(path_in_repo=ruta_remota, path_or_fileobj=ruta_trabajo.read_bytes())
                )

        if operaciones:
            api.create_commit(
                repo_id=dataset_id,
                repo_type="dataset",
                operations=operaciones,
                commit_message=f"chore: sincronizar {len(operaciones)} ficheros parquet locales previos",
            )
            logging.info(f"[HF] Sincronizados {len(operaciones)} ficheros locales previos hacia {dataset_id}.")
    return len(operaciones)

def archivar_tabla_d1_a_hf(api: HfApi, dataset_id: str, base_url: str, sync_token: str, tabla: str) -> int:
    """Descarga lotes pendientes de una tabla D1, fusiona en Parquet en Hugging Face y purga D1."""
    cfg = CONFIG_TABLAS[tabla]
    sub = cfg["subcarpeta"]
    prefijo = cfg["prefijo"]
    dedup = cfg["dedup"]
    total_procesadas = 0

    while True:
        resp = requests.get(
            f"{base_url}/api/export",
            params={"token": sync_token, "table": tabla},
            timeout=60,
        )
        resp.raise_for_status()
        filas = resp.json().get("rows", list())
        if not filas:
            break

        max_id = max(int(r["id"]) for r in filas)
        df = pd.DataFrame(filas)
        dfs_por_dia: dict[str, list[pd.DataFrame]] = dict()

        if tabla in ("emt_llegadas", "meteo_madrid"):
            df["fecha_dia"] = df["timestamp_captura_utc"].str.slice(0, 10).str.replace("-", "")
            cols = [c for c in df.columns if c not in ("id", "fecha_dia")]
            for dia, grupo in df.groupby("fecha_dia"):
                dfs_por_dia.setdefault(str(dia), list()).append(grupo[cols])

        elif tabla == "trafico_xml_raw":
            for _, fila in df.iterrows():
                ts_utc = str(fila["timestamp_captura_utc"])
                dia = ts_utc[:10].replace("-", "")
                df_sensores = _parsear_xml_trafico_gzip(str(fila["xml_gzip_base64"]), ts_utc)
                if not df_sensores.empty:
                    dfs_por_dia.setdefault(dia, list()).append(df_sensores)

        operaciones = list()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for dia, lista_dfs in sorted(dfs_por_dia.items()):
                df_lote_dia = pd.concat(lista_dfs, ignore_index=True)
                nombre_fichero = f"{prefijo}_{dia}.parquet"
                ruta_remota = f"realtime/{sub}/{nombre_fichero}"
                ruta_trabajo = tmp_path / nombre_fichero

                _descargar_parquet_remoto_si_existe(api, dataset_id, ruta_remota, ruta_trabajo)
                filas_finales = _anexar_parquet(df_lote_dia, ruta_trabajo, dedup)
                logging.info(f"[{tabla}] {ruta_remota} actualizado ({filas_finales:,} filas totales).")

                operaciones.append(
                    CommitOperationAdd(path_in_repo=ruta_remota, path_or_fileobj=ruta_trabajo.read_bytes())
                )

            if operaciones:
                ts_commit = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                api.create_commit(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    operations=operaciones,
                    commit_message=f"data({sub}): archivar lote {tabla} ({ts_commit})",
                )

        # Purgar en Cloudflare D1 únicamente tras confirmar el commit en Hugging Face
        resp_purge = requests.post(
            f"{base_url}/api/purge",
            params={"token": sync_token},
            json={"table": tabla, "max_id": max_id},
            timeout=30,
        )
        resp_purge.raise_for_status()
        total_procesadas += len(filas)
        logging.info(f"[{tabla}] Lote archivado en Hugging Face y purgado en D1 (hasta ID {max_id}).")

    return total_procesadas


def ejecutar_archivado_nube() -> None:
    """Punto de entrada principal del archivado automático Cloudflare D1 -> Hugging Face."""
    base_url = os.getenv("CLOUDFLARE_WORKER_URL", "").strip().rstrip("/")
    sync_token = os.getenv("SYNC_SECRET_TOKEN", "").strip()
    if not base_url or not sync_token:
        raise ValueError("Faltan CLOUDFLARE_WORKER_URL o SYNC_SECRET_TOKEN en las variables de entorno.")

    api, dataset_id = _obtener_api_hf()
    logging.info(f"Iniciando archivado Cloud-to-Cloud hacia Dataset privado: {dataset_id}")

    sincronizar_locales_previos_a_hf(api, dataset_id)

    for tabla in ("emt_llegadas", "trafico_xml_raw", "meteo_madrid"):
        n = archivar_tabla_d1_a_hf(api, dataset_id, base_url, sync_token, tabla)
        logging.info(f"Resumen [{tabla}]: {n} registros archivados en Hugging Face.")


if __name__ == "__main__":
    ejecutar_archivado_nube()
