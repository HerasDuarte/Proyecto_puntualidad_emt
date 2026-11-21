"""
Descargador Incremental desde Hugging Face Dataset Privado -> Entorno Local.
Recupera todos los ficheros .parquet diarios consolidados (realtime/emt/, realtime/trafico/,
realtime/meteorologia/) y las instantáneas oficiales GTFS (gtfs_snapshots/gtfs_YYYYMMDD.zip)
almacenadas en el Dataset privado de Hugging Face para reproducibilidad y entrenamiento ML.
"""

import logging
import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from src.config import (
    DIR_RAW,
    DIR_VERSIONES_GTFS_RAW,
    HF_DATASET_ID_DEFECTO,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _obtener_cliente_hf() -> tuple[HfApi, str]:
    """Inicializa el cliente de Hugging Face y valida el token de acceso."""
    hf_token = os.getenv("HF_TOKEN", "").strip()
    dataset_id = os.getenv("HF_DATASET_ID", HF_DATASET_ID_DEFECTO).strip() or HF_DATASET_ID_DEFECTO
    if not hf_token:
        raise ValueError("Falta HF_TOKEN en las variables de entorno o archivo .env")
    return HfApi(token=hf_token), dataset_id


def descargar_historico_desde_hf(forzar_actualizacion: bool = False) -> dict[str, int]:
    """
    Descarga de forma incremental los .parquet diarios de realtime/ y los snapshots
    gtfs_snapshots/ desde Hugging Face hacia data/raw/.
    Retorna un diccionario con el conteo de ficheros descargados/actualizados por categoría.
    """
    api, dataset_id = _obtener_cliente_hf()
    archivos_remotos = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")

    parquets_remotos = sorted(
        f for f in archivos_remotos
        if f.startswith("realtime/") and f.endswith(".parquet")
    )
    snapshots_gtfs_remotos = sorted(
        f for f in archivos_remotos
        if f.startswith("gtfs_snapshots/") and (f.endswith(".zip") or f.endswith(".json"))
    )

    resumen = {"parquets_actualizados": 0, "snapshots_gtfs_actualizados": 0}

    # Siempre refrescamos los ficheros más recientes de cada subcarpeta por si recibieron nuevos lotes
    ultimos_por_carpeta: dict[str, str] = dict()
    for ruta_r in parquets_remotos:
        carpeta = str(Path(ruta_r).parent)
        ultimos_por_carpeta[carpeta] = ruta_r

    for ruta_remota in parquets_remotos:
        ruta_local = DIR_RAW / Path(ruta_remota)
        es_ultimo_dia = ultimos_por_carpeta.get(str(Path(ruta_remota).parent)) == ruta_remota

        if ruta_local.exists() and not es_ultimo_dia and not forzar_actualizacion:
            continue

        ruta_cache = hf_hub_download(
            repo_id=dataset_id,
            filename=ruta_remota,
            repo_type="dataset",
            token=api.token,
        )
        ruta_local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ruta_cache, ruta_local)
        resumen["parquets_actualizados"] += 1
        logging.info(f"[HF -> Local] Sincronizado {ruta_local}")

    DIR_VERSIONES_GTFS_RAW.mkdir(parents=True, exist_ok=True)
    for ruta_remota in snapshots_gtfs_remotos:
        nombre_fichero = Path(ruta_remota).name
        ruta_local = DIR_VERSIONES_GTFS_RAW / nombre_fichero

        if ruta_local.exists() and not nombre_fichero.endswith(".json") and not forzar_actualizacion:
            continue

        ruta_cache = hf_hub_download(
            repo_id=dataset_id,
            filename=ruta_remota,
            repo_type="dataset",
            token=api.token,
        )
        shutil.copy2(ruta_cache, ruta_local)
        resumen["snapshots_gtfs_actualizados"] += 1
        logging.info(f"[HF -> Local] Snapshot GTFS descargado en {ruta_local}")

    logging.info(
        f"Sincronización histórica completada: {resumen['parquets_actualizados']} Parquets y "
        f"{resumen['snapshots_gtfs_actualizados']} artefactos GTFS actualizados en data/raw/."
    )
    return resumen


if __name__ == "__main__":
    descargar_historico_desde_hf()
