"""
Orquestador de Ingesta Sincronizada Multi-Fuente en Tiempo Real.
Coordina mediante relojes independientes la captura continua de:
  1. Llegadas en tiempo real de autobuses EMT Madrid (cada 3 min, lotes de 25 paradas,
     con tope autónomo de 4.500 peticiones y auto-reactivación a las 24h + 1 min).
  2. Estado del tráfico urbano del Ayuntamiento de Madrid (cada 5 min).
  3. Condiciones meteorológicas de Madrid vía Open-Meteo (cada 15 min).

Los datos se almacenan incrementalmente por fecha en formato Parquet bajo data/raw/realtime/.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import time

# Garantizar que la raíz del proyecto esté en sys.path al ejecutar directamente o con -m
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import pandas as pd

from src.emt_realtime import ClienteEMTRealtime
from src.meteorologia import obtener_meteorologia_actual
from src.trafico import obtener_trafico_saneado

# Intervalos de captura en segundos
INTERVALO_EMT_SEG = 180       # 3 minutos (25 paradas/lote -> ~500 reqs/hora)
INTERVALO_TRAFICO_SEG = 300   # 5 minutos (sincronizado con actualización XML Ayto. Madrid)
INTERVALO_METEO_SEG = 900     # 15 minutos (alineado con ventana tramo_15min)
TAMANO_LOTE_EMT = 25

DIR_REALTIME = Path("data/raw/realtime")
DIR_LOGS = Path("logs")


def configurar_logging() -> None:
    """Configura salida simultánea por consola y fichero de registro en logs/."""
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    ruta_log = DIR_LOGS / "ingesta_realtime.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(ruta_log, encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True
    )


def anexar_a_parquet_diario(df_nuevo: pd.DataFrame, subcarpeta: str, prefijo: str) -> None:
    """
    Anexa un lote de registros nuevos al fichero Parquet particionado por fecha UTC
    correspondiente al día actual, creando la estructura si no existe.
    """
    if df_nuevo.empty:
        return

    fecha_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    dir_destino = DIR_REALTIME / subcarpeta
    dir_destino.mkdir(parents=True, exist_ok=True)
    ruta_parquet = dir_destino / f"{prefijo}_{fecha_str}.parquet"

    if ruta_parquet.exists():
        df_existente = pd.read_parquet(ruta_parquet)
        df_combinado = pd.concat([df_existente, df_nuevo], ignore_index=True)
    else:
        df_combinado = df_nuevo

    df_combinado.to_parquet(ruta_parquet, index=False, compression="zstd")


def ejecutar_orquestador() -> None:
    """Bucle principal del orquestador multi-fuente con parada segura mediante Ctrl+C."""
    configurar_logging()
    logging.info("======================================================================")
    logging.info("INICIANDO ORQUESTADOR DE INGESTA EN TIEMPO REAL (EMT + TRAFICO + METEO)")
    logging.info("Presiona Ctrl+C en cualquier momento para detener de forma segura.")
    logging.info("======================================================================")

    cliente_emt = ClienteEMTRealtime()

    ultima_ejecucion_emt = 0.0
    ultima_ejecucion_trafico = 0.0
    ultima_ejecucion_meteo = 0.0

    try:
        while True:
            ahora_mono = time.monotonic()

            # 1. RELOJ EMT MADRID (cada 3 minutos)
            if (ahora_mono - ultima_ejecucion_emt) >= INTERVALO_EMT_SEG or ultima_ejecucion_emt == 0.0:
                try:
                    df_emt, estado_emt = cliente_emt.ejecutar_lote_rotativo(tamano_lote=TAMANO_LOTE_EMT)
                    anexar_a_parquet_diario(df_emt, subcarpeta="emt", prefijo="emt_llegadas")
                    logging.info(
                        f"[EMT] {estado_emt} | Filas capturadas en lote: {len(df_emt):,}"
                    )
                except Exception as exc:
                    logging.error(f"[EMT] Error en ciclo de captura: {exc}")
                ultima_ejecucion_emt = time.monotonic()

            # 2. RELOJ TRÁFICO AYUNTAMIENTO DE MADRID (cada 5 minutos)
            if (ahora_mono - ultima_ejecucion_trafico) >= INTERVALO_TRAFICO_SEG or ultima_ejecucion_trafico == 0.0:
                try:
                    df_traf = obtener_trafico_saneado()
                    df_traf["timestamp_captura_utc"] = datetime.now(timezone.utc).isoformat()
                    anexar_a_parquet_diario(df_traf, subcarpeta="trafico", prefijo="trafico_sensores")
                    logging.info(
                        f"[TRÁFICO] Captura completada | Sensores operativos guardados: {len(df_traf):,}"
                    )
                except Exception as exc:
                    logging.error(f"[TRÁFICO] Error en ciclo de captura: {exc}")
                ultima_ejecucion_trafico = time.monotonic()

            # 3. RELOJ METEOROLOGÍA OPEN-METEO (cada 15 minutos)
            if (ahora_mono - ultima_ejecucion_meteo) >= INTERVALO_METEO_SEG or ultima_ejecucion_meteo == 0.0:
                try:
                    df_meteo = obtener_meteorologia_actual()
                    anexar_a_parquet_diario(df_meteo, subcarpeta="meteo", prefijo="meteo_madrid")
                    logging.info("[METEO] Instantánea climática de 15 min guardada correctamente.")
                except Exception as exc:
                    logging.error(f"[METEO] Error en ciclo de captura: {exc}")
                ultima_ejecucion_meteo = time.monotonic()

            time.sleep(5)

    except KeyboardInterrupt:
        logging.info("======================================================================")
        logging.info("Detención manual solicitada (Ctrl+C). Todos los lotes han sido guardados.")
        logging.info("======================================================================")


if __name__ == "__main__":
    ejecutar_orquestador()
