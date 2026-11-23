"""
Archivador Cloud-to-Cloud: Cloudflare D1 -> Hugging Face Dataset Privado.
1. Sube a Hugging Face cualquier fichero .parquet local existente en data/raw/realtime/
   que aún no esté consolidado en el dataset remoto.
2. Descarga los lotes pendientes de Cloudflare D1 (emt_llegadas, trafico_xml_raw, meteo_madrid),
   los fusiona sin duplicados con los .parquet diarios del Dataset privado en Hugging Face,
   los sube y purga los registros procesados en Cloudflare D1.
3. Audita diariamente las 8 tablas operativas del paquete oficial gtfs_emt.zip (SHA-256)
   y archiva una copia íntegra en gtfs_snapshots/gtfs_YYYYMMDD.zip dentro de Hugging Face
   cada vez que EMT Madrid publique cambios de calendario (Navidad, Año Nuevo, etc.).
"""

import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    CONFIG_TABLAS_REALTIME as CONFIG_TABLAS,
)
from src.config import (
    DIR_REALTIME as DIR_REALTIME_LOCAL,
)
from src.config import (
    RUTA_HISTORIAL_GTFS_HF,
)
from src.utils import (
    anexar_parquet_deduplicado as _anexar_parquet,
)
from src.utils import (
    calcular_hashes_operativos_zip as _calcular_hashes_operativos_zip,
)
from src.utils import (
    descargar_zip_gtfs_oficial as _descargar_zip_gtfs_oficial,
)
from src.utils import (
    obtener_cliente_hf as _obtener_api_hf,
)
from src.utils import (
    parsear_xml_trafico_gzip as _parsear_xml_trafico_gzip,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _descargar_fichero_remoto_si_existe(api: HfApi, dataset_id: str, ruta_remota: str, ruta_destino: Path) -> bool:
    """Descarga un fichero desde Hugging Face si ya existe en el repositorio remoto."""
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


def respaldar_snapshot_gtfs_en_hf(api: HfApi, dataset_id: str) -> bool:
    """
    Audita diariamente las 8 tablas operativas de gtfs_emt.zip.
    Si detecta cambios respecto al último registro guardado en Hugging Face,
    sube gtfs_snapshots/gtfs_YYYYMMDD.zip y actualiza historial_gtfs_nube.json.
    """
    try:
        zip_bytes = _descargar_zip_gtfs_oficial()
    except Exception as exc:
        logging.warning(f"[HF-GTFS] No se pudo descargar gtfs_emt.zip hoy ({exc}).")
        return False

    hashes_remotos = _calcular_hashes_operativos_zip(zip_bytes)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ruta_hist_tmp = Path(tmp_dir) / "historial_gtfs_nube.json"
        historial = list()
        if _descargar_fichero_remoto_si_existe(api, dataset_id, RUTA_HISTORIAL_GTFS_HF, ruta_hist_tmp):
            historial = json.loads(ruta_hist_tmp.read_text(encoding="utf-8"))

        hashes_previos = historial[-1].get("hashes_tablas", dict()) if historial else dict()
        tablas_cambiadas = [
            tabla for tabla, h in hashes_remotos.items()
            if hashes_previos.get(tabla) != h
        ]

        ahora_utc = datetime.now(timezone.utc).isoformat()
        if not tablas_cambiadas:
            etiqueta_vigente = historial[-1]["version_tag"]
            historial[-1]["ultima_verificacion_utc"] = ahora_utc
            api.upload_file(
                path_or_fileobj=json.dumps(
                    historial, indent=2, ensure_ascii=False
                ).encode("utf-8"),
                path_in_repo=RUTA_HISTORIAL_GTFS_HF,
                repo_id=dataset_id,
                repo_type="dataset",
                commit_message=(
                    f"chore(gtfs): verificación diaria sin cambios ({etiqueta_vigente})"
                ),
            )
            logging.info(
                f"[HF-GTFS] Sin cambios en las 8 tablas operativas "
                f"(vigente en nube: {etiqueta_vigente})."
            )
            return False

        etiqueta_hoy = datetime.now(timezone.utc).strftime("%Y%m%d")
        ruta_zip_remoto = f"gtfs_snapshots/gtfs_{etiqueta_hoy}.zip"
        historial.append({
            "version_tag": etiqueta_hoy,
            "detectado_utc": ahora_utc,
            "ultima_verificacion_utc": ahora_utc,
            "tablas_modificadas": tablas_cambiadas,
            "hashes_tablas": hashes_remotos,
            "ruta_zip": ruta_zip_remoto,
        })


        operaciones = [
            CommitOperationAdd(path_in_repo=ruta_zip_remoto, path_or_fileobj=zip_bytes),
            CommitOperationAdd(
                path_in_repo=RUTA_HISTORIAL_GTFS_HF,
                path_or_fileobj=json.dumps(historial, indent=2, ensure_ascii=False).encode("utf-8"),
            ),
        ]
        api.create_commit(
            repo_id=dataset_id,
            repo_type="dataset",
            operations=operaciones,
            commit_message=f"data(gtfs): archivar snapshot oficial gtfs_{etiqueta_hoy}.zip ({len(tablas_cambiadas)} tablas actualizadas)",
        )
        logging.info(f"[HF-GTFS] Nuevo snapshot archivado en {ruta_zip_remoto} (cambios: {', '.join(tablas_cambiadas)}).")
        return True

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

                if _descargar_fichero_remoto_si_existe(api, dataset_id, ruta_remota, ruta_trabajo):
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

                _descargar_fichero_remoto_si_existe(api, dataset_id, ruta_remota, ruta_trabajo)
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

    respaldar_snapshot_gtfs_en_hf(api, dataset_id)


if __name__ == "__main__":
    ejecutar_archivado_nube()
