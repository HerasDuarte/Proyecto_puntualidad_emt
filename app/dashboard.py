import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (  # noqa: E402
    HF_DATASET_ID_DEFECTO,
    RUTA_HISTORIAL_GTFS_HF,
    RUTA_HISTORIAL_GTFS_LOCAL,
)

load_dotenv()

URL_API_LIVE = os.getenv(
    "URL_API_LIVE_EMT",
    "https://orquestador-emt-madrid.herasduarte.workers.dev/api/live",
)
ZONA_MADRID = ZoneInfo("Europe/Madrid")
HEADERS_HTTP = {
    "User-Agent": "Mozilla/5.0 ProyectoPuntualidadEMT/1.0",
    "Accept": "application/json",
}

st.set_page_config(
    page_title="Monitor Puntualidad EMT Madrid",
    page_icon="🚌",
    layout="wide",
)


def formatear_hora_madrid(ts_iso: str | None) -> str:
    """Convierte un timestamp ISO UTC a hora local de Madrid legible."""
    if not ts_iso:
        return "Sin datos"
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.astimezone(ZONA_MADRID).strftime("%d/%m/%Y %H:%M:%S")
    except ValueError:
        return ts_iso


@st.cache_data(ttl=120)
def cargar_datos_en_vivo(url: str) -> dict[str, Any]:
    """Consulta el endpoint /api/live de Cloudflare Workers con caché de 2 minutos."""
    resp = requests.get(url, headers=HEADERS_HTTP, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def cargar_historial_gtfs() -> tuple[list[dict[str, Any]], str]:
    """Carga el historial GTFS desde Hugging Face o, en su defecto, desde local."""
    dataset_id = (
        os.getenv("HF_DATASET_ID", HF_DATASET_ID_DEFECTO).strip()
        or HF_DATASET_ID_DEFECTO
    )
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    try:
        ruta_remota = hf_hub_download(
            repo_id=dataset_id,
            filename=RUTA_HISTORIAL_GTFS_HF,
            repo_type="dataset",
            token=hf_token,
        )
        datos = json.loads(Path(ruta_remota).read_text(encoding="utf-8"))
        return datos, "Hugging Face Hub (nube)"
    except (EntryNotFoundError, HfHubHTTPError):
        if RUTA_HISTORIAL_GTFS_LOCAL.exists():
            datos = json.loads(RUTA_HISTORIAL_GTFS_LOCAL.read_text(encoding="utf-8"))
            return datos, "Repositorio local (pendiente de primera sincronización diaria en nube)"
        return [], "Sin registros todavía"


# --- Cabecera principal ---
col_titulo, col_boton = st.columns(2)
with col_titulo:
    st.title("🚌 Monitor de Puntualidad y Movilidad — EMT Madrid")
    st.caption(
        "Ingesta autónoma 24/7 (Cloudflare Workers + D1) · Lakehouse (Hugging Face) · "
        "Powered by EMT de Madrid"
    )
with col_boton:
    st.write("")
    if st.button("🔄 Actualizar ahora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

tab_live, tab_gtfs = st.tabs(
    ["📡 Estado en Tiempo Real (Madrid)", "🗂️ Auditoría y Versiones GTFS"]
)

# --- Pestaña 1: Tiempo real ---
with tab_live:
    try:
        datos = cargar_datos_en_vivo(URL_API_LIVE)
        meteo = datos.get("meteo_actual") or {}
        trafico = datos.get("trafico_actual") or {}
        cuota = datos.get("estado_cuota_emt") or {}
        top_lineas = [
            fila
            for fila in (datos.get("top_lineas_espera") or [])
            if (fila.get("espera_media_min") or 0) < 180
        ]
        llegadas = datos.get("ultimas_llegadas") or []

        st.info(
            f"Última consulta al servidor: **{formatear_hora_madrid(datos.get('generado_utc'))}** "
            f"(hora local de Madrid)"
        )

        k1, k2, k3, k4 = st.columns(4)
        temp = meteo.get("temperatura_c")
        lluvia = meteo.get("lluvia_mm")
        k1.metric(
            "🌡️ Temperatura / Lluvia",
            f"{temp:.1f} °C" if temp is not None else "N/D",
            f"Lluvia: {lluvia:.1f} mm" if lluvia is not None else "Sin precipitación",
        )
        k2.metric(
            "🚦 Último pulso Tráfico",
            formatear_hora_madrid(trafico.get("timestamp_captura_utc")).split(" ")[-1],
            f"{round((trafico.get('tamano_bytes_original') or 0) / 1024, 1)} KB XML",
        )
        k3.metric(
            "📡 Peticiones EMT hoy",
            f"{cuota.get('peticiones_realizadas', 0):,}".replace(",", "."),
            f"Parada índice: {cuota.get('indice_parada_actual', 0)}",
        )
        peor_linea = next(iter(top_lineas), {})
        k4.metric(
            "⏱️ Mayor espera media",
            f"Línea {peor_linea.get('line', '—')}",
            f"{peor_linea.get('espera_media_min', 0)} min (máx {peor_linea.get('espera_max_min', 0)} min)",
        )

        st.divider()

        col_graf, col_tabla = st.columns([1.1, 1.2])
        with col_graf:
            st.subheader("Líneas con mayor tiempo de espera (últimos ~25 min)")
            if top_lineas:
                df_top = pd.DataFrame(top_lineas)
                fig = px.bar(
                    df_top,
                    x="espera_media_min",
                    y="line",
                    orientation="h",
                    text="espera_media_min",
                    labels={"espera_media_min": "Espera media (min)", "line": "Línea EMT"},
                    color="espera_media_min",
                    color_continuous_scale="YlOrRd",
                )
                fig.update_layout(
                    yaxis={"type": "category", "categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    height=380,
                    margin={"l": 10, "r": 10, "t": 10, "b": 10},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aún no hay suficientes lecturas recientes sin código centinela.")

        with col_tabla:
            st.subheader("Últimas 25 llegadas capturadas en directo")
            if llegadas:
                df_lleg = pd.DataFrame(llegadas)
                df_lleg["hora_madrid"] = df_lleg["timestamp_captura_utc"].apply(
                    lambda x: formatear_hora_madrid(x).split(" ")[-1]
                )
                df_lleg["espera_min"] = df_lleg["estimate_arrive_sec"].apply(
                    lambda s: "Sin estimación" if s >= 90000 else f"{round(s / 60.0, 1)} min"
                )
                vista = df_lleg[
                    ["hora_madrid", "line", "stop_id", "destination", "espera_min", "distance_bus_m"]
                ].rename(
                    columns={
                        "hora_madrid": "Hora",
                        "line": "Línea",
                        "stop_id": "Parada",
                        "destination": "Destino",
                        "espera_min": "Espera",
                        "distance_bus_m": "Distancia (m)",
                    }
                )
                st.dataframe(vista, use_container_width=True, height=380, hide_index=True)
            else:
                st.warning("No hay llegadas recientes disponibles.")

    except Exception as exc:
        st.error(f"No se pudo conectar con el servicio en tiempo real ({exc}).")

# --- Pestaña 2: Auditoría GTFS ---
with tab_gtfs:
    st.subheader("Control de Versiones Oficiales del Catálogo GTFS (EMT Madrid)")
    try:
        historial, origen_hist = cargar_historial_gtfs()
        st.caption(f"Origen de datos: **{origen_hist}**")
        if historial:
            vigente = historial[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("📦 Versión GTFS vigente", vigente.get("version_tag", "N/D"))
            c2.metric(
                "📅 Fecha de detección",
                formatear_hora_madrid(vigente.get("detectado_utc")),
            )
            c3.metric(
                "🔍 Última verificación",
                formatear_hora_madrid(
                    vigente.get("ultima_verificacion_utc") or vigente.get("detectado_utc")
                ),
            )

            filas_hist = []
            for item in reversed(historial):
                filas_hist.append(
                    {
                        "Versión": item.get("version_tag"),
                        "Detectado (Madrid)": formatear_hora_madrid(item.get("detectado_utc")),
                        "Última revisión": formatear_hora_madrid(
                            item.get("ultima_verificacion_utc") or item.get("detectado_utc")
                        ),
                        "Tablas modificadas": ", ".join(item.get("tablas_modificadas") or []),
                        "Snapshot ZIP": item.get("ruta_zip", "Inicial"),
                    }
                )
            st.dataframe(pd.DataFrame(filas_hist), use_container_width=True, hide_index=True)
        else:
            st.info(
                "Aún no hay registros en el historial. Aparecerán automáticamente tras "
                "la próxima ejecución programada."
            )
    except Exception as exc:
        st.warning(f"No se pudo cargar el historial ({exc}).")
