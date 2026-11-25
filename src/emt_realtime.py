"""
Módulo de Ingesta en Tiempo Real de Llegadas EMT Madrid (API MobilityLabs v2).
Implementa barrido rotativo estratificado (Round-Robin) sobre las 4.928 paradas EMT,
renovación automática de token y control autónomo de cuota diaria sincronizado con
el reinicio UTC (00:00 UTC / 01:00 CET) con tope de seguridad de 4.500 peticiones/día.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils import HEADERS_HTTP_PROYECTO

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_URL_EMT = "https://" + "openapi." + "emtmadrid.es/v2"
RUTA_STOPS_PROCESADO = Path("data/processed/stops_limpio.csv")
RUTA_ESTADO_CUOTA = Path("data/raw/realtime/estado_cuota_emt.json")

LIMITE_PETICIONES_CICLO = 4500


def construir_lista_paradas_estratificada(ruta_stops: Path | None = None) -> list[str]:
    """
    Construye la secuencia rotativa de stop_id intercalando de forma equilibrada
    paradas del Interior de la M-30 y de la Periferia Exterior.
    """
    ruta_stops = ruta_stops or RUTA_STOPS_PROCESADO
    if not ruta_stops.exists():
        raise FileNotFoundError(f"Falta el fichero de paradas procesado: {ruta_stops}")
    df_stops = pd.read_csv(ruta_stops, dtype={"stop_id": str})
    m30 = df_stops.loc[df_stops["en_alta_densidad_real"], "stop_id"].tolist()
    periferia = df_stops.loc[~df_stops["en_alta_densidad_real"], "stop_id"].tolist()

    secuencia = list()
    max_len = max(len(m30), len(periferia))
    for i in range(max_len):
        if i < len(m30):
            secuencia.append(m30[i])
        if i < len(periferia):
            secuencia.append(periferia[i])
    return secuencia


def _cargar_estado_cuota(ruta_estado: Path | None = None) -> dict:
    """Lee el estado persistente de cuota diaria UTC y puntero de rotación desde disco."""
    ruta_estado = ruta_estado or RUTA_ESTADO_CUOTA
    ahora_utc = datetime.now(timezone.utc)
    hoy_utc_str = ahora_utc.strftime("%Y-%m-%d")

    if ruta_estado.exists():
        with open(ruta_estado, "r", encoding="utf-8") as f:
            estado = json.load(f)
        if estado.get("fecha_cuota_utc") != hoy_utc_str:
            estado["fecha_cuota_utc"] = hoy_utc_str
            estado["session_id"] = f"SES_{ahora_utc.strftime('%Y%m%d_%H%M%S')}"
            estado["session_start_utc"] = ahora_utc.isoformat()
            estado["last_request_utc"] = None
            estado["next_allowed_start_utc"] = None
            estado["peticiones_realizadas"] = 0
            estado["en_pausa_cuota"] = False
            _guardar_estado_cuota(estado, ruta_estado)
        return estado

    return {
        "fecha_cuota_utc": hoy_utc_str,
        "session_id": f"SES_{ahora_utc.strftime('%Y%m%d_%H%M%S')}",
        "session_start_utc": ahora_utc.isoformat(),
        "last_request_utc": None,
        "next_allowed_start_utc": None,
        "peticiones_realizadas": 0,
        "indice_parada_actual": 0,
        "en_pausa_cuota": False
    }


def _guardar_estado_cuota(estado: dict, ruta_estado: Path | None = None) -> None:
    """Persiste el estado actual de cuota y rotación en formato JSON."""
    ruta_estado = ruta_estado or RUTA_ESTADO_CUOTA
    ruta_estado.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_estado, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, ensure_ascii=False)


class ClienteEMTRealtime:
    """Gestor autónomo de consultas en tiempo real a MobilityLabs EMT Madrid."""

    def __init__(self, limite_ciclo: int = LIMITE_PETICIONES_CICLO):
        self.client_id = os.getenv("EMT_CLIENT_ID", "").strip()
        self.passkey = os.getenv("EMT_PASSKEY", "").strip()
        if not self.client_id or not self.passkey:
            raise ValueError("Faltan EMT_CLIENT_ID o EMT_PASSKEY en el archivo .env")

        self.limite_ciclo = limite_ciclo
        self.access_token = None
        self.paradas_rotativas = construir_lista_paradas_estratificada()
        self.estado = _cargar_estado_cuota()

    def _autenticar(self) -> None:
        """Obtiene un nuevo accessToken temporal desde el endpoint de login."""
        url_login = f"{BASE_URL_EMT}/mobilitylabs/user/login/"
        headers = {
            "X-ClientId": self.client_id,
            "passKey": self.passkey,
            "User-Agent": HEADERS_HTTP_PROYECTO["User-Agent"],
            "Accept": "application/json"
        }
        resp = requests.get(url_login, headers=headers, timeout=15)
        resp.raise_for_status()
        datos = resp.json()

        if datos.get("code") not in ("00", "01"):
            raise RuntimeError(f"Error de autenticación MobilityLabs: {datos}")

        bloque = next(iter(datos.get("data", list())), dict())
        token = bloque.get("accessToken")
        if not token:
            raise RuntimeError(f"Login EMT sin accessToken válido: {datos}")
        self.access_token = token
        logging.info("Token de sesión MobilityLabs EMT renovado correctamente.")

    def _consultar_parada(self, stop_id: str) -> list[dict]:
        """Consulta el endpoint de llegadas en tiempo real para una parada concreta."""
        if not self.access_token:
            self._autenticar()

        url_arrives = f"{BASE_URL_EMT}/transport/busemtmad/stops/{stop_id}/arrives/"
        headers = {
            "accessToken": self.access_token,
            "Content-Type": "application/json",
            "User-Agent": HEADERS_HTTP_PROYECTO["User-Agent"]
        }
        payload = {
            "cultureInfo": "ES",
            "Text_StopRequired_YN": "N",
            "Text_EstimationsRequired_YN": "Y",
            "Text_IncidencesRequired_YN": "N",
            "DateTime_Referenced_Incidencies_YYYYMMDD": ""
        }

        resp = requests.post(url_arrives, headers=headers, json=payload, timeout=25)

        if resp.status_code == 401:
            self._autenticar()
            headers["accessToken"] = self.access_token
            resp = requests.post(url_arrives, headers=headers, json=payload, timeout=25)

        resp.raise_for_status()
        datos = resp.json()

        if datos.get("code") == "80":
            self._autenticar()
            headers["accessToken"] = self.access_token
            resp = requests.post(url_arrives, headers=headers, json=payload, timeout=25)
            resp.raise_for_status()
            datos = resp.json()

        bloque = next(iter(datos.get("data", list())), dict())
        return bloque.get("Arrive", list())

    def ejecutar_lote_rotativo(self, tamano_lote: int = 25) -> Tuple[pd.DataFrame, str]:
        """
        Ejecuta un lote rotativo de consultas respetando el tope diario de 4.500
        peticiones y reiniciando automáticamente al cambiar de día UTC (00:01 UTC).
        Retorna (DataFrame con llegadas capturadas, estado_operativo).
        """
        ahora_utc = datetime.now(timezone.utc)
        hoy_utc_str = ahora_utc.strftime("%Y-%m-%d")

        if self.estado.get("fecha_cuota_utc") != hoy_utc_str:
            logging.info(f"Nuevo día UTC detectado ({hoy_utc_str}). Reiniciando contador diario a 0/{self.limite_ciclo}.")
            self.estado["fecha_cuota_utc"] = hoy_utc_str
            self.estado["session_id"] = f"SES_{ahora_utc.strftime('%Y%m%d_%H%M%S')}"
            self.estado["session_start_utc"] = ahora_utc.isoformat()
            self.estado["last_request_utc"] = None
            self.estado["next_allowed_start_utc"] = None
            self.estado["peticiones_realizadas"] = 0
            self.estado["en_pausa_cuota"] = False
            _guardar_estado_cuota(self.estado)

        if self.estado.get("en_pausa_cuota"):
            siguiente_dia_utc = (ahora_utc + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
            return pd.DataFrame(), f"STANDBY_CUOTA (Reinicio automático a las {siguiente_dia_utc.strftime('%Y-%m-%d %H:%M')} UTC)"

        restantes = self.limite_ciclo - self.estado["peticiones_realizadas"]
        if restantes <= 0:
            return pd.DataFrame(), "STANDBY_CUOTA"

        n_consultas = min(tamano_lote, restantes)
        total_paradas = len(self.paradas_rotativas)
        if total_paradas == 0:
            raise RuntimeError("La lista estratificada de paradas está vacía.")
        registros_llegadas = list()

        for _ in range(n_consultas):
            idx = self.estado["indice_parada_actual"] % total_paradas
            stop_id = self.paradas_rotativas[idx]
            ts_consulta = datetime.now(timezone.utc)

            try:
                llegadas = self._consultar_parada(stop_id)
                for bus in llegadas:
                    registros_llegadas.append({
                        "session_id": self.estado["session_id"],
                        "session_start_utc": self.estado["session_start_utc"],
                        "session_end_utc": None,
                        "timestamp_captura_utc": ts_consulta.isoformat(),
                        "stop_id": str(stop_id),
                        "line": str(bus.get("line", "")),
                        "destination": str(bus.get("destination", "")),
                        "bus_id": str(bus.get("bus", "")),
                        "estimate_arrive_sec": pd.to_numeric(bus.get("estimateArrive"), errors="coerce"),
                        "distance_bus_m": pd.to_numeric(bus.get("DistanceBus"), errors="coerce"),
                        "position_type_bus": bus.get("positionTypeBus"),
                        "is_head": bus.get("isHead")
                    })
            except Exception as exc:
                logging.warning(f"Error consultando parada {stop_id}: {exc}")

            self.estado["peticiones_realizadas"] += 1
            self.estado["indice_parada_actual"] = (idx + 1) % total_paradas
            self.estado["last_request_utc"] = ts_consulta.isoformat()

        if self.estado["peticiones_realizadas"] >= self.limite_ciclo:
            reactivacion_dt = (ahora_utc + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
            self.estado["en_pausa_cuota"] = True
            self.estado["next_allowed_start_utc"] = reactivacion_dt.isoformat()
            logging.warning(
                f"Tope diario alcanzado ({self.limite_ciclo} peticiones). "
                f"EMT en pausa automática hasta {reactivacion_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC."
            )

        _guardar_estado_cuota(self.estado)

        df_lote = pd.DataFrame(registros_llegadas)
        if not df_lote.empty:
            df_lote["session_end_utc"] = self.estado["last_request_utc"]

        resumen_estado = (
            f"ACTIVO ({self.estado['peticiones_realizadas']}/{self.limite_ciclo} reqs | "
            f"Índice parada: {self.estado['indice_parada_actual']}/{total_paradas})"
        )
        return df_lote, resumen_estado
