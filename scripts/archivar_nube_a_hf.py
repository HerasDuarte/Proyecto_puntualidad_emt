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

import io
import json
import logging
import os
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generar_catalogo_dashboard import (
    RUTA_CATALOGO_HF,
    construir_catalogo_dashboard,
)
from src.config import (
    CONFIG_TABLAS_REALTIME as CONFIG_TABLAS,
)
from src.config import (
    DIR_REALTIME as DIR_REALTIME_LOCAL,
)
from src.config import (
    RUTA_HISTORIAL_GTFS_HF,
    RUTA_TENSION_DIARIA_HF,
)
from src.utils import (
    HEADERS_HTTP_PROYECTO,
    ZONA_MADRID,
    auditar_cuarentenas_gtfs_zip,
    comparar_zips_gtfs,
    extraer_contexto_madrid,
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
            force_download=True,
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
    if not hashes_remotos:
        logging.warning("[HF-GTFS] El ZIP descargado no contiene tablas operativas válidas.")
        return False

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
        if not tablas_cambiadas and historial:
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

        zip_prev_bytes: bytes | None = None
        ruta_zip_prev = historial[-1].get("ruta_zip") if historial else None
        if ruta_zip_prev:
            ruta_prev_tmp = Path(tmp_dir) / "gtfs_prev.zip"
            if _descargar_fichero_remoto_si_existe(
                api, dataset_id, str(ruta_zip_prev), ruta_prev_tmp
            ) and zipfile.is_zipfile(ruta_prev_tmp):
                zip_prev_bytes = ruta_prev_tmp.read_bytes()

        resumen_cambios = comparar_zips_gtfs(zip_prev_bytes, zip_bytes)
        cuarentenas_gtfs = auditar_cuarentenas_gtfs_zip(zip_bytes)

        etiqueta_hoy = datetime.now(timezone.utc).strftime("%Y%m%d")
        ruta_zip_remoto = f"gtfs_snapshots/gtfs_{etiqueta_hoy}.zip"
        historial.append({
            "version_tag": etiqueta_hoy,
            "detectado_utc": ahora_utc,
            "ultima_verificacion_utc": ahora_utc,
            "tablas_modificadas": tablas_cambiadas,
            "hashes_tablas": hashes_remotos,
            "ruta_zip": ruta_zip_remoto,
            "resumen_cambios": resumen_cambios,
            "cuarentenas": cuarentenas_gtfs,
        })

        operaciones = [
            CommitOperationAdd(path_in_repo=ruta_zip_remoto, path_or_fileobj=zip_bytes),
            CommitOperationAdd(
                path_in_repo=RUTA_HISTORIAL_GTFS_HF,
                path_or_fileobj=json.dumps(
                    historial, indent=2, ensure_ascii=False
                ).encode("utf-8"),
            ),
        ]

        # Regenerar automáticamente el catálogo ligero del dashboard con el nuevo GTFS
        dir_gtfs_tmp = Path(tmp_dir) / "gtfs_nuevo"
        dir_cat_tmp = Path(tmp_dir) / "catalogo_nuevo"
        dir_gtfs_tmp.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf_in:
                for miembro in zf_in.namelist():
                    nombre_txt = Path(miembro).name
                    if nombre_txt.endswith(".txt"):
                        (dir_gtfs_tmp / nombre_txt).write_bytes(zf_in.read(miembro))
            archivos_cat = construir_catalogo_dashboard(
                dir_gtfs=dir_gtfs_tmp, dir_salida=dir_cat_tmp
            )
            for ruta_pq in archivos_cat.values():
                operaciones.append(
                    CommitOperationAdd(
                        path_in_repo=f"{RUTA_CATALOGO_HF}/{ruta_pq.name}",
                        path_or_fileobj=ruta_pq.read_bytes(),
                    )
                )
        except Exception as exc:
            logging.warning(
                "[HF-GTFS] No se pudo regenerar catalogo_dashboard automático (%s).",
                exc,
            )
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


def actualizar_resumen_tension_diaria_hf(
    api: HfApi,
    dataset_id: str,
    df_emt: pd.DataFrame,
) -> int:
    """Cruza las llegadas EMT del día con GTFS y acumula anomalías (exceso >= 4 min) en HF."""
    if df_emt.empty:
        return 0

    df_val = df_emt.copy()
    df_val["estimate_arrive_sec"] = pd.to_numeric(
        df_val["estimate_arrive_sec"], errors="coerce"
    )
    df_val["line"] = df_val["line"].astype(str).str.strip()
    df_val["stop_id"] = df_val["stop_id"].astype(str).str.strip()
    df_val["destination"] = df_val["destination"].fillna("").astype(str).str.strip()
    df_val = df_val.loc[
        df_val["estimate_arrive_sec"].notna()
        & (df_val["estimate_arrive_sec"] >= 0)
        & (df_val["estimate_arrive_sec"] < 90000)
        & (df_val["line"] != "")
    ].copy()
    if df_val.empty:
        return 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ruta_freq = tmp_path / "frecuencias_oficiales.parquet"
        ruta_par = tmp_path / "paradas.parquet"

        if not _descargar_fichero_remoto_si_existe(
            api, dataset_id, f"{RUTA_CATALOGO_HF}/frecuencias_oficiales.parquet", ruta_freq
        ):
            logging.info("[HF-TENSION] Sin frecuencias_oficiales.parquet en HF; se omite resumen.")
            return 0
        _descargar_fichero_remoto_si_existe(
            api, dataset_id, f"{RUTA_CATALOGO_HF}/paradas.parquet", ruta_par
        )

        df_freq = (
            pd.read_parquet(ruta_freq)
            .groupby(["line", "tipo_dia", "hora"], as_index=False)["headway_min"]
            .median()
        )
        df_par = (
            pd.read_parquet(ruta_par)[["stop_id", "stop_name"]]
            if ruta_par.exists()
            else pd.DataFrame(columns=["stop_id", "stop_name"])
        )

        fechas_mad = (
            pd.to_datetime(df_val["timestamp_captura_utc"], utc=True, errors="coerce")
            .dt.tz_convert(ZONA_MADRID)
            .dt.strftime("%Y-%m-%d")
        )
        ctx = df_val["timestamp_captura_utc"].apply(extraer_contexto_madrid)
        df_val["fecha_madrid"] = fechas_mad
        df_val["tipo_dia"] = [c[0] for c in ctx]
        df_val["hora"] = [c[1] for c in ctx]

        df_agg = (
            df_val.dropna(subset=["fecha_madrid"])
            .groupby(
                ["fecha_madrid", "tipo_dia", "hora", "line", "stop_id", "destination"],
                as_index=False,
            )
            .agg(
                lecturas=("estimate_arrive_sec", "count"),
                espera_media_min=("estimate_arrive_sec", lambda s: round(float(s.mean()) / 60.0, 1)),
                espera_max_min=("estimate_arrive_sec", lambda s: round(float(s.max()) / 60.0, 1)),
            )
        )
        df_cruce = df_agg.merge(
            df_freq, on=["line", "tipo_dia", "hora"], how="left"
        ).merge(df_par, on="stop_id", how="left")
        df_cruce["exceso_min"] = (
            df_cruce["espera_media_min"] - df_cruce["headway_min"]
        ).round(1)
        df_anom = df_cruce.loc[
            df_cruce["headway_min"].notna() & (df_cruce["exceso_min"] >= 4.0)
        ].copy()
        if df_anom.empty:
            return 0

        ruta_resumen = tmp_path / "resumen_tension_diaria.parquet"
        _descargar_fichero_remoto_si_existe(
            api, dataset_id, RUTA_TENSION_DIARIA_HF, ruta_resumen
        )
        total = _anexar_parquet(
            df_anom,
            ruta_resumen,
            ["fecha_madrid", "hora", "line", "stop_id", "destination"],
        )
        api.upload_file(
            path_or_fileobj=str(ruta_resumen),
            path_in_repo=RUTA_TENSION_DIARIA_HF,
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=f"data(analytics): actualiza resumen diario de tensión ({len(df_anom)} filas)",
        )
        logging.info(
            "[HF-TENSION] +%d anomalías registradas en %s (total acumulado: %d).",
            len(df_anom),
            RUTA_TENSION_DIARIA_HF,
            total,
        )
        return len(df_anom)


def archivar_tabla_d1_a_hf(api: HfApi, dataset_id: str, base_url: str, sync_token: str, tabla: str) -> int:
    """Descarga lotes pendientes de una tabla D1, fusiona en Parquet en Hugging Face y purga D1."""
    cfg = CONFIG_TABLAS[tabla]
    sub = cfg["subcarpeta"]
    prefijo = cfg["prefijo"]
    dedup = cfg["dedup"]
    total_procesadas = 0
    lotes_emt: list[pd.DataFrame] = list()

    while True:
        resp = requests.get(
            f"{base_url}/api/export",
            params={"table": tabla},
            headers={**HEADERS_HTTP_PROYECTO, "X-Sync-Token": sync_token},
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
            if tabla == "emt_llegadas":
                lotes_emt.append(df[cols])
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
            json={"table": tabla, "max_id": max_id},
            headers={**HEADERS_HTTP_PROYECTO, "X-Sync-Token": sync_token},
            timeout=30,
        )
        resp_purge.raise_for_status()
        total_procesadas += len(filas)
        logging.info(f"[{tabla}] Lote archivado en Hugging Face y purgado en D1 (hasta ID {max_id}).")

    if tabla == "emt_llegadas" and lotes_emt:
        try:
            actualizar_resumen_tension_diaria_hf(
                api, dataset_id, pd.concat(lotes_emt, ignore_index=True)
            )
        except Exception as exc:
            logging.warning("[HF-TENSION] Error generando resumen diario (%s).", exc)

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
