import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from pyproj import Transformer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (  # noqa: E402
    DIR_PROCESSED,
    HF_DATASET_ID_DEFECTO,
    RUTA_HISTORIAL_GTFS_HF,
    RUTA_HISTORIAL_GTFS_LOCAL,
)
from src.utils import (  # noqa: E402
    HEADERS_HTTP_PROYECTO,
    calcular_distancia_haversine,
    extraer_contexto_madrid,
    formatear_hora_madrid,
    parsear_xml_trafico_gzip,
)

load_dotenv()

URL_API_BASE = os.getenv(
    "URL_API_BASE_EMT",
    "https://orquestador-emt-madrid.herasduarte.workers.dev",
).rstrip("/")
URL_API_LIVE = os.getenv("URL_API_LIVE_EMT", f"{URL_API_BASE}/api/live")
URL_API_TRAFICO_LIVE = os.getenv(
    "URL_API_TRAFICO_LIVE", f"{URL_API_BASE}/api/trafico-live"
)
URL_API_PARADA = os.getenv("URL_API_PARADA", f"{URL_API_BASE}/api/parada")
URL_API_TENSION = os.getenv("URL_API_TENSION", f"{URL_API_BASE}/api/tension-hoy")

DIR_CATALOGO_LOCAL = DIR_PROCESSED / "catalogo_dashboard"
USA_SCATTERMAP = hasattr(go, "Scattermap")
TRANSFORMADOR_UTM_A_WGS84 = Transformer.from_crs(
    "EPSG:25830", "EPSG:4326", always_xy=True
)
ESTILO_OSM_GRIS: dict[str, Any] = {
    "version": 8,
    "sources": {
        "osm": {
            "type": "raster",
            "tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            "tileSize": 256,
            "attribution": "© OpenStreetMap contributors",
        }
    },
    "layers": [
        {
            "id": "osm-escala-grises",
            "type": "raster",
            "source": "osm",
            "paint": {"raster-saturation": -1},
        }
    ],
}

st.set_page_config(
    page_title="Monitor Puntualidad EMT Madrid",
    page_icon="🚌",
    layout="wide",
)


def obtener_config_hf() -> tuple[str, str | None]:
    """Obtiene dataset ID y token de Hugging Face desde entorno o secrets."""
    dataset_id = os.getenv("HF_DATASET_ID", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not dataset_id or not hf_token:
        try:
            dataset_id = dataset_id or str(
                st.secrets.get("HF_DATASET_ID", "")
            ).strip()
            hf_token = hf_token or str(st.secrets.get("HF_TOKEN", "")).strip()
        except Exception:
            pass
    return (dataset_id or HF_DATASET_ID_DEFECTO), (hf_token or None)


@st.cache_data(ttl=120)
def cargar_datos_en_vivo(url: str) -> dict[str, Any]:
    """Consulta el endpoint /api/live de Cloudflare Workers (caché 2 min)."""
    resp = requests.get(url, headers=HEADERS_HTTP_PROYECTO, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=120)
def consultar_parada_en_vivo(url_base: str, stop_id: str) -> dict[str, Any]:
    """Consulta /api/parada?id=XXXX con caché local de 2 minutos."""
    resp = requests.get(
        url_base, params={"id": stop_id}, headers=HEADERS_HTTP_PROYECTO, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def cargar_tension_hoy(url: str) -> dict[str, Any]:
    """Consulta /api/tension-hoy con las mayores esperas reales del día."""
    resp = requests.get(url, headers=HEADERS_HTTP_PROYECTO, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def cargar_trafico_en_vivo(url: str) -> tuple[pd.DataFrame, str | None]:
    """Descarga el último boletín XML GZIP de D1 y proyecta UTM a lat/lon."""
    resp = requests.get(url, headers=HEADERS_HTTP_PROYECTO, timeout=20)
    resp.raise_for_status()
    payload = resp.json().get("trafico") or {}
    xml_b64 = payload.get("xml_gzip_base64")
    ts_utc = payload.get("timestamp_captura_utc")
    if not xml_b64 or not ts_utc:
        return pd.DataFrame(), ts_utc

    df = parsear_xml_trafico_gzip(xml_b64, ts_utc)
    if df.empty:
        return df, ts_utc

    df = df.dropna(subset=["st_x", "st_y", "carga"]).copy()
    lons, lats = TRANSFORMADOR_UTM_A_WGS84.transform(
        df["st_x"].to_numpy(), df["st_y"].to_numpy()
    )
    df["lon"] = lons
    df["lat"] = lats
    filtro_madrid = df["lat"].between(40.25, 40.65) & df["lon"].between(
        -3.95, -3.45
    )
    return df.loc[filtro_madrid].copy(), ts_utc


@st.cache_data(ttl=3600)
def cargar_catalogo_gtfs() -> dict[str, pd.DataFrame]:
    """Carga los 5 Parquet ligeros desde disco local o Hugging Face Hub."""
    nombres = (
        "paradas",
        "lineas",
        "recorridos_paradas",
        "trazados",
        "frecuencias_oficiales",
    )
    catalogo: dict[str, pd.DataFrame] = {}
    dataset_id, hf_token = obtener_config_hf()
    for nombre in nombres:
        archivo = f"{nombre}.parquet"
        ruta_local = DIR_CATALOGO_LOCAL / archivo
        if ruta_local.exists():
            catalogo[nombre] = pd.read_parquet(ruta_local)
            continue
        ruta_hf = hf_hub_download(
            repo_id=dataset_id,
            filename=f"catalogo_dashboard/{archivo}",
            repo_type="dataset",
            token=hf_token,
        )
        catalogo[nombre] = pd.read_parquet(ruta_hf)
    return catalogo


@st.cache_data(ttl=300)
def cargar_historial_gtfs() -> tuple[list[dict[str, Any]], str]:
    """Carga el historial GTFS desde Hugging Face o desde repositorio local."""
    dataset_id, hf_token = obtener_config_hf()
    try:
        ruta_remota = hf_hub_download(
            repo_id=dataset_id,
            filename=RUTA_HISTORIAL_GTFS_HF,
            repo_type="dataset",
            token=hf_token,
            force_download=True,
        )
        datos = json.loads(Path(ruta_remota).read_text(encoding="utf-8"))
        return datos, "Hugging Face Hub (nube)"
    except (EntryNotFoundError, HfHubHTTPError):
        if RUTA_HISTORIAL_GTFS_LOCAL.exists():
            datos = json.loads(
                RUTA_HISTORIAL_GTFS_LOCAL.read_text(encoding="utf-8")
            )
            return (
                datos,
                "Repositorio local (pendiente de sincronización en nube)",
            )
        return [], "Sin registros todavía"


def configurar_layout_mapa(
    fig: go.Figure,
    lat: float,
    lon: float,
    zoom: float = 11.7,
    estilo: str | dict[str, Any] | None = None,
    altura: int = 520,
) -> None:
    """Aplica configuración de mapa minimalista compatible con Plotly 5 y 6."""
    estilo_final = estilo or (
        ESTILO_OSM_GRIS if USA_SCATTERMAP else "open-street-map"
    )
    cfg_mapa = {
        "style": estilo_final,
        "center": {"lat": lat, "lon": lon},
        "zoom": zoom,
    }
    if USA_SCATTERMAP:
        fig.update_layout(map=cfg_mapa)
    else:
        fig.update_layout(mapbox=cfg_mapa)
    fig.update_layout(height=altura, margin={"l": 0, "r": 0, "t": 10, "b": 0})


def traza_mapa(**kwargs: Any) -> Any:
    """Construye Scattermap o Scattermapbox según la versión de Plotly."""
    return (
        go.Scattermap(**kwargs) if USA_SCATTERMAP else go.Scattermapbox(**kwargs)
    )


def traza_densidad_mapa(**kwargs: Any) -> Any:
    """Construye Densitymap o Densitymapbox según la versión de Plotly."""
    return (
        go.Densitymap(**kwargs) if USA_SCATTERMAP else go.Densitymapbox(**kwargs)
    )


# --- Carga compartida inicial ---
datos_live: dict[str, Any] = {}
error_live: str | None = None
try:
    datos_live = cargar_datos_en_vivo(URL_API_LIVE)
except Exception as exc:
    error_live = str(exc)

df_traf_global = pd.DataFrame()
ts_traf_global: str | None = None
error_traf_global: str | None = None
try:
    df_traf_global, ts_traf_global = cargar_trafico_en_vivo(
        URL_API_TRAFICO_LIVE
    )
except Exception as exc:
    error_traf_global = str(exc)

catalogo: dict[str, pd.DataFrame] = {}
error_catalogo: str | None = None
try:
    catalogo = cargar_catalogo_gtfs()
except Exception as exc:
    error_catalogo = str(exc)

# --- Cabecera principal ---
col_titulo, col_boton = st.columns((4, 1))
with col_titulo:
    st.title("🚌 Monitor de Puntualidad y Movilidad — EMT Madrid")
    st.caption(
        "Ingesta autónoma 24/7 (Cloudflare Workers + D1) · Lakehouse "
        "(Hugging Face) · Powered by EMT de Madrid"
    )
with col_boton:
    st.write("")
    if st.button("🔄 Actualizar ahora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

tab_viajero, tab_trafico, tab_salud = st.tabs(
    (
        "🚌 Tu Parada y Línea (Tiempo Real)",
        "🚦 Tráfico en Vivo (Madrid)",
        "🛠️ Salud del Sistema y Auditoría GTFS",
    )
)

# --- Pestaña 1: Experiencia útil para el viajero EMT ---
with tab_viajero:
    if error_catalogo:
        st.error(f"No se pudo cargar el catálogo GTFS ({error_catalogo}).")
    else:
        df_paradas = catalogo["paradas"]
        df_lineas = catalogo["lineas"]
        df_recorridos = catalogo["recorridos_paradas"]
        df_trazados = catalogo["trazados"]
        df_frecuencias = catalogo["frecuencias_oficiales"]

        meteo = datos_live.get("meteo_actual") or {}
        temp_act = meteo.get("temperatura_c")
        lluvia_act = meteo.get("lluvia_mm")
        viento_act = meteo.get("viento_velocidad_kmh")

        # --- Bloque A: Consulta de parada y ruta ---
        st.subheader("Consulta Tu Parada, Línea y Estado del Trayecto")
        df_paradas_con_linea = df_paradas.loc[
            df_paradas["lineas"].str.len() > 0
        ].copy()
        df_paradas_con_linea["etiqueta"] = (
            df_paradas_con_linea["stop_id"]
            + " — "
            + df_paradas_con_linea["stop_name"]
            + " (Líneas: "
            + df_paradas_con_linea["lineas"]
            + ")"
        )
        etiquetas = df_paradas_con_linea["etiqueta"].tolist()
        idx_defecto = next(
            (i for i, e in enumerate(etiquetas) if e.startswith("923 —")),
            0,
        )

        col_sel, col_btn = st.columns((3.2, 1.0))
        with col_sel:
            parada_elegida = st.selectbox(
                "Busca tu parada por número, calle o línea:",
                etiquetas,
                index=idx_defecto,
            )
        stop_sel = parada_elegida.split(" — ", 1)[0].strip()
        fila_parada = df_paradas_con_linea.loc[
            df_paradas_con_linea["stop_id"] == stop_sel
        ].iloc[0]

        with col_btn:
            st.write("")
            st.write("")
            if st.button(
                "🔍 Ver llegadas en vivo",
                type="primary",
                use_container_width=True,
            ):
                try:
                    st.session_state["ultima_parada_id"] = stop_sel
                    st.session_state["ultima_parada_resp"] = (
                        consultar_parada_en_vivo(URL_API_PARADA, stop_sel)
                    )
                except Exception as exc:
                    st.error(f"Error al consultar la parada ({exc}).")

        # Contexto inmediato de clima y tráfico alrededor de la parada
        carga_entorno = None
        sensores_cerca = 0
        if not df_traf_global.empty:
            dists = calcular_distancia_haversine(
                float(fila_parada["stop_lat"]),
                float(fila_parada["stop_lon"]),
                df_traf_global["lat"].to_numpy(),
                df_traf_global["lon"].to_numpy(),
            )
            cercanos = df_traf_global.loc[dists <= 600]
            sensores_cerca = len(cercanos)
            if sensores_cerca > 0:
                carga_entorno = float(cercanos["carga"].mean())

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "📍 Parada seleccionada",
            f"Parada {stop_sel}",
            str(fila_parada["stop_name"])[:40],
        )
        m2.metric(
            "🚦 Tráfico entorno (< 600 m)",
            f"{carga_entorno:.0f} % carga"
            if carga_entorno is not None
            else "Sin sensores < 600 m",
            f"{sensores_cerca} sensores activos cerca",
        )
        m3.metric(
            "🌡️ Clima en Madrid ahora",
            f"{temp_act:.1f} °C" if temp_act is not None else "N/D",
            (
                f"Lluvia: {lluvia_act:.1f} mm · Viento: "
                f"{viento_act or 0:.0f} km/h"
            )
            if lluvia_act is not None
            else "Sin datos recientes",
        )

        # Resultado de llegadas en directo si el usuario consultó esta parada
        if st.session_state.get("ultima_parada_id") == stop_sel:
            resp_par = st.session_state.get("ultima_parada_resp") or {}
            if resp_par.get("cupo_web_agotado"):
                st.warning(
                    "⚠️ **Cupo diario de consultas web en vivo agotado "
                    f"({resp_par.get('peticiones_web_hoy', 150)} / "
                    f"{resp_par.get('limite_web_diario', 150)}).** "
                    "Este dato no se volverá a actualizar en tiempo real hasta "
                    "el reinicio de la cuota diaria (`00:00 UTC`). Se muestra "
                    "la última lectura guardada por el recolector automático."
                )
            else:
                st.success(
                    f"Lectura en directo (**{formatear_hora_madrid(resp_par.get('timestamp_lectura_utc'))}**"
                    f" · Consultas web hoy: "
                    f"**{resp_par.get('peticiones_web_hoy', 0)} / "
                    f"{resp_par.get('limite_web_diario', 150)}**)"
                )

            lleg_par = resp_par.get("llegadas") or []
            if lleg_par:
                df_lp = pd.DataFrame(lleg_par)
                df_lp["Llegada"] = df_lp["estimate_arrive_sec"].apply(
                    lambda s: "En parada" if s < 45 else f"{round(s / 60, 1)} min"
                )
                df_lp["Cabecera"] = df_lp["is_head"].apply(
                    lambda x: "Sí" if str(x).strip().lower() in ("true", "1") else "En ruta"
                )
                vista_lp = df_lp[
                    ["line", "destination", "Llegada", "distance_bus_m", "Cabecera"]
                ].rename(
                    columns={
                        "line": "Línea",
                        "destination": "Destino",
                        "distance_bus_m": "Distancia (m)",
                    }
                )
                st.dataframe(
                    vista_lp, use_container_width=True, hide_index=True
                )
            else:
                st.info(
                    "No hay estimaciones de llegada disponibles ahora mismo "
                    "para esta parada."
                )

        st.divider()

        # --- Planificador visual de línea, destino y tráfico en ruta ---
        st.subheader("Trazado de Tu Línea, Tráfico en Ruta y Destino")
        lineas_de_parada = [
            x.strip()
            for x in str(fila_parada["lineas"]).split(",")
            if x.strip()
        ]
        todas_lineas = sorted(df_lineas["line"].unique().tolist())
        opciones_linea = lineas_de_parada + [
            ln for ln in todas_lineas if ln not in lineas_de_parada
        ]

        c_l1, c_l2, c_l3 = st.columns((1.0, 1.3, 1.5))
        with c_l1:
            linea_sel = st.selectbox("Línea EMT:", opciones_linea)

        sentidos_linea = df_lineas.loc[df_lineas["line"] == linea_sel].copy()
        sentidos_linea["desc_sentido"] = (
            "Sentido "
            + sentidos_linea["direction_id"].astype(str)
            + " → "
            + sentidos_linea["headsign"]
        )
        # Preferir sentido que contenga la parada seleccionada
        idx_sentido = 0
        for i, row_s in enumerate(sentidos_linea.itertuples()):
            tiene_parada = not df_recorridos.loc[
                (df_recorridos["line"] == linea_sel)
                & (df_recorridos["direction_id"] == row_s.direction_id)
                & (df_recorridos["stop_id"] == stop_sel)
            ].empty
            if tiene_parada:
                idx_sentido = i
                break

        with c_l2:
            sentido_lbl = st.selectbox(
                "Sentido / Destino:",
                sentidos_linea["desc_sentido"].tolist(),
                index=idx_sentido,
            )
        fila_sentido = sentidos_linea.loc[
            sentidos_linea["desc_sentido"] == sentido_lbl
        ].iloc[0]
        dir_sel = int(fila_sentido["direction_id"])
        shape_sel = str(fila_sentido["shape_id"])

        df_rec_linea = df_recorridos.loc[
            (df_recorridos["line"] == linea_sel)
            & (df_recorridos["direction_id"] == dir_sel)
        ].sort_values("stop_sequence")

        # Determinar paradas destino posibles
        origen_en_rec = df_rec_linea.loc[df_rec_linea["stop_id"] == stop_sel]
        if not origen_en_rec.empty:
            seq_orig = int(origen_en_rec.iloc[0]["stop_sequence"])
            min_orig = float(origen_en_rec.iloc[0]["min_desde_cabecera"])
            dist_orig_m = int(origen_en_rec.iloc[0]["dist_m"])
            df_destinos = df_rec_linea.loc[
                df_rec_linea["stop_sequence"] > seq_orig
            ].copy()
        else:
            seq_orig = -1
            min_orig = 0.0
            dist_orig_m = 0
            df_destinos = df_rec_linea.copy()

        opciones_dest = ["(Solo ver recorrido completo)"] + (
            df_destinos["stop_id"] + " — " + df_destinos["stop_name"]
        ).tolist()
        with c_l3:
            dest_lbl = st.selectbox(
                "Parada donde te bajas (opcional):", opciones_dest
            )

        # Frecuencia oficial ahora mismo para la línea elegida
        tipo_hoy, hora_hoy = extraer_contexto_madrid(None)
        freq_linea = df_frecuencias.loc[
            (df_frecuencias["line"] == linea_sel)
            & (df_frecuencias["direction_id"] == dir_sel)
            & (df_frecuencias["tipo_dia"] == tipo_hoy)
            & (df_frecuencias["hora"] == hora_hoy)
        ]
        headway_act = (
            float(freq_linea["headway_min"].iloc[0])
            if not freq_linea.empty
            else None
        )

        r1, r2, r3 = st.columns(3)
        r1.metric(
            "⏱ Frecuencia oficial GTFS ahora",
            f"Cada {headway_act:.1f} min"
            if headway_act is not None
            else "Según horario fijo",
            f"Tipo día: {tipo_hoy} · {hora_hoy:02d}:00 h",
        )
        if not dest_lbl.startswith("("):
            stop_dest = dest_lbl.split(" — ", 1)[0].strip()
            fila_dest = df_rec_linea.loc[
                df_rec_linea["stop_id"] == stop_dest
            ].iloc[0]
            duracion_min = max(
                1.0, float(fila_dest["min_desde_cabecera"]) - min_orig
            )
            n_paradas_tramo = (
                int(fila_dest["stop_sequence"]) - seq_orig
                if seq_orig >= 0
                else int(fila_dest["stop_sequence"])
            )
            dist_km = max(0.1, (int(fila_dest["dist_m"]) - dist_orig_m) / 1000)
            r2.metric(
                "🕒 Tiempo teórico de trayecto",
                f"~{duracion_min:.0f} min",
                f"{n_paradas_tramo} paradas · {dist_km:.1f} km",
            )
        else:
            stop_dest = None
            r2.metric(
                "🚏 Paradas del recorrido",
                f"{len(df_rec_linea)}",
                str(fila_sentido["route_long_name"])[:40],
            )

        # Sensores de tráfico próximos al corredor de la línea (< 450 m)
        sensores_corredor = pd.DataFrame()
        if not df_traf_global.empty and not df_rec_linea.empty:
            lat_rec = df_rec_linea["stop_lat"].to_numpy()
            lon_rec = df_rec_linea["stop_lon"].to_numpy()
            lat_tr = df_traf_global["lat"].to_numpy()
            lon_tr = df_traf_global["lon"].to_numpy()
            min_dist = np.full(len(df_traf_global), np.inf)
            for la, lo in zip(lat_rec, lon_rec, strict=True):
                d = calcular_distancia_haversine(float(la), float(lo), lat_tr, lon_tr)
                min_dist = np.minimum(min_dist, d)
            sensores_corredor = df_traf_global.loc[
                (min_dist <= 450) & (df_traf_global["carga"] >= 20)
            ].sort_values("carga", ascending=True)

        carga_corredor = (
            float(sensores_corredor["carga"].mean())
            if not sensores_corredor.empty
            else None
        )
        r3.metric(
            "🛣️ Tráfico en tu recorrido (< 450 m)",
            f"{carga_corredor:.0f} % carga media"
            if carga_corredor is not None
            else "Fluido (< 20 %)",
            f"{len(sensores_corredor)} sensores en el entorno",
        )

        fig_ruta = go.Figure()

        # 1. Capa inferior: Mapa de calor de congestión a lo largo del corredor
        if not sensores_corredor.empty:
            fig_ruta.add_trace(
                traza_densidad_mapa(
                    lat=sensores_corredor["lat"],
                    lon=sensores_corredor["lon"],
                    z=sensores_corredor["carga"].clip(0, 100),
                    radius=24,
                    opacity=0.75,
                    colorscale=[
                        [0.0, "rgba(16, 185, 129, 0.0)"],
                        [0.25, "#10B981"],
                        [0.55, "#F59E0B"],
                        [0.85, "#EF4444"],
                        [1.0, "#991B1B"],
                    ],
                    colorbar={
                        "title": "Carga (%)",
                        "len": 0.65,
                        "y": 0.5,
                        "x": 1.02,
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # 2. Capa intermedia: Recorrido oficial de la línea (ribete blanco + línea azul)
        df_trazo = df_trazados.loc[
            df_trazados["shape_id"] == shape_sel
        ].sort_values("seq")
        if not df_trazo.empty:
            fig_ruta.add_trace(
                traza_mapa(
                    lat=df_trazo["lat"],
                    lon=df_trazo["lon"],
                    mode="lines",
                    line={"width": 6, "color": "#FFFFFF"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig_ruta.add_trace(
                traza_mapa(
                    lat=df_trazo["lat"],
                    lon=df_trazo["lon"],
                    mode="lines",
                    line={"width": 4, "color": "#2563EB"},
                    name=f"Ruta Línea {linea_sel}",
                    hoverinfo="skip",
                )
            )

        # 3. Capa superior: Paradas con estilo Citymapper (círculo blanco + aro azul marino)
        if not df_rec_linea.empty:
            fig_ruta.add_trace(
                traza_mapa(
                    lat=df_rec_linea["stop_lat"],
                    lon=df_rec_linea["stop_lon"],
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": "#FFFFFF",
                        "opacity": 1.0,
                    },
                    text=(
                        df_rec_linea["stop_id"]
                        + " — "
                        + df_rec_linea["stop_name"]
                        + " (Min: +"
                        + df_rec_linea["min_desde_cabecera"].astype(str)
                        + ")"
                    ),
                    hoverinfo="text",
                    name="Paradas intermedias",
                )
            )
            fig_ruta.add_trace(
                traza_mapa(
                    lat=df_rec_linea["stop_lat"],
                    lon=df_rec_linea["stop_lon"],
                    mode="markers",
                    marker={
                        "size": 5,
                        "color": "#1E3A8A",
                        "opacity": 1.0,
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # 4. Parada origen destacada en verde esmeralda con ribete blanco
        fig_ruta.add_trace(
            traza_mapa(
                lat=[float(fila_parada["stop_lat"])],
                lon=[float(fila_parada["stop_lon"])],
                mode="markers",
                marker={"size": 17, "color": "#FFFFFF", "opacity": 1.0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig_ruta.add_trace(
            traza_mapa(
                lat=[float(fila_parada["stop_lat"])],
                lon=[float(fila_parada["stop_lon"])],
                mode="markers",
                marker={"size": 13, "color": "#059669", "opacity": 1.0},
                text=[f"Tu parada (Origen): {stop_sel} — {fila_parada['stop_name']}"],
                hoverinfo="text",
                name="Tu parada (Origen)",
            )
        )

        # 5. Parada destino destacada en rojo carmesí con ribete blanco
        if stop_dest:
            fila_d_map = df_rec_linea.loc[
                df_rec_linea["stop_id"] == stop_dest
            ].iloc[0]
            fig_ruta.add_trace(
                traza_mapa(
                    lat=[float(fila_d_map["stop_lat"])],
                    lon=[float(fila_d_map["stop_lon"])],
                    mode="markers",
                    marker={"size": 17, "color": "#FFFFFF", "opacity": 1.0},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig_ruta.add_trace(
                traza_mapa(
                    lat=[float(fila_d_map["stop_lat"])],
                    lon=[float(fila_d_map["stop_lon"])],
                    mode="markers",
                    marker={"size": 13, "color": "#DC2626", "opacity": 1.0},
                    text=[f"Tu destino: {stop_dest} — {fila_d_map['stop_name']}"],
                    hoverinfo="text",
                    name="Tu destino",
                )
            )

        lat_c = (
            float(df_rec_linea["stop_lat"].mean())
            if not df_rec_linea.empty
            else float(fila_parada["stop_lat"])
        )
        lon_c = (
            float(df_rec_linea["stop_lon"].mean())
            if not df_rec_linea.empty
            else float(fila_parada["stop_lon"])
        )
        fig_ruta.update_layout(
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.01,
                "xanchor": "left",
                "x": 0,
            }
        )
        configurar_layout_mapa(fig_ruta, lat_c, lon_c, zoom=12.4, altura=460)
        st.plotly_chart(fig_ruta, use_container_width=True)

        st.divider()

        # --- Bloque B: Semáforo de Líneas Tensionadas HOY + Histórico ---
        st.subheader("🚦 Semáforo de Líneas Tensionadas HOY y Recurrencia Histórica")
        st.caption(
            "Cruza las esperas reales medidas hoy frente a la frecuencia oficial GTFS, "
            "agregando además la recurrencia histórica de paradas reincidentes."
        )
        try:
            # 1. Cargar histórico acumulado de tensión desde HF si existe
            ds_id_th, hf_tok_th = obtener_config_hf()
            df_historico_ten = pd.DataFrame()
            try:
                ruta_hist_ten = hf_hub_download(
                    repo_id=ds_id_th,
                    filename="analytics/resumen_tension_diaria.parquet",
                    repo_type="dataset",
                    token=hf_tok_th,
                )
                df_historico_ten = pd.read_parquet(ruta_hist_ten)
            except Exception:
                pass

            # 2. Datos de hoy desde el Worker
            datos_ten = cargar_tension_hoy(URL_API_TENSION)
            regs = datos_ten.get("registros") or []
            if not regs and df_historico_ten.empty:
                st.info("Aún no hay suficientes capturas ni histórico de tensión disponible.")
            else:
                if regs:
                    df_ten = pd.DataFrame(regs)
                    df_ten["stop_id"] = df_ten["stop_id"].astype(str).str.strip()
                    df_ten["line"] = df_ten["line"].astype(str).str.strip()

                    ctx_list = df_ten["ultima_captura_utc"].apply(
                        extraer_contexto_madrid
                    )
                    df_ten["tipo_dia"] = [c[0] for c in ctx_list]
                    df_ten["hora"] = [c[1] for c in ctx_list]

                    df_freq_agg = (
                        df_frecuencias.groupby(
                            ["line", "tipo_dia", "hora"], as_index=False
                        )["headway_min"]
                        .median()
                    )
                    df_ten = df_ten.merge(
                        df_freq_agg, on=["line", "tipo_dia", "hora"], how="left"
                    ).merge(
                        df_paradas[["stop_id", "stop_name"]],
                        on="stop_id",
                        how="left",
                    )
                    df_ten["exceso_min"] = (
                        df_ten["espera_media_min"] - df_ten["headway_min"]
                    ).round(1)

                    df_anom = df_ten.dropna(subset=["headway_min"]).copy()
                    if not df_anom.empty:
                        def etiquetar_semaforo(exc: float) -> str:
                            if exc >= 10:
                                return "🔴 Alta (+10 min)"
                            if exc >= 4:
                                return "🟠 Moderada (+4 a +10 min)"
                            return "🟢 En rango habitual"

                        df_anom["Estado"] = df_anom["exceso_min"].apply(etiquetar_semaforo)
                        df_anom["Parada"] = (
                            df_anom["stop_id"] + " — " + df_anom["stop_name"].fillna("Parada EMT")
                        )
                        st.markdown("##### 📌 Instantánea de Anomalías Registradas HOY")
                        vista_ten = df_anom.sort_values("exceso_min", ascending=False).head(20)[
                            [
                                "Estado",
                                "line",
                                "Parada",
                                "destination",
                                "headway_min",
                                "espera_media_min",
                                "exceso_min",
                            ]
                        ].rename(
                            columns={
                                "line": "Línea",
                                "destination": "Destino",
                                "headway_min": "Frecuencia GTFS",
                                "espera_media_min": "Espera media",
                                "exceso_min": "Exceso (min)",
                            }
                        )
                        st.dataframe(vista_ten, use_container_width=True, hide_index=True)

                # 3. Ranking de recurrencia histórica si existe acumulado
                if not df_historico_ten.empty:
                    st.markdown("##### 📈 Ranking Histórico de Reincidencia (Paradas más tensionadas)")
                    df_rec_agg = (
                        df_historico_ten.groupby(
                            ["line", "stop_id", "destination"], as_index=False
                        )
                        .agg(
                            dias_afectados=("fecha_madrid", "nunique"),
                            exceso_historico_medio=("exceso_min", "mean"),
                        )
                        .sort_values(["dias_afectados", "exceso_historico_medio"], ascending=[False, False])
                        .merge(df_paradas[["stop_id", "stop_name"]], on="stop_id", how="left")
                    )
                    df_rec_agg["Parada"] = df_rec_agg["stop_id"] + " — " + df_rec_agg["stop_name"].fillna("Parada EMT")
                    df_rec_agg["Exceso medio"] = df_rec_agg["exceso_historico_medio"].round(1)
                    vista_rec = df_rec_agg.head(15)[
                        ["dias_afectados", "line", "Parada", "destination", "Exceso medio"]
                    ].rename(
                        columns={
                            "dias_afectados": "Días reincidentes",
                            "line": "Línea",
                            "destination": "Destino",
                        }
                    )
                    st.dataframe(vista_rec, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"No se pudo calcular el semáforo de tensión ({exc}).")

# --- Pestaña 2: Tráfico en vivo (Madrid) ---
with tab_trafico:
    st.subheader(
        "Mapa de Congestión del Tráfico en Tiempo Real (Ayuntamiento de Madrid)"
    )
    if error_traf_global:
        st.error(
            f"No se pudo cargar el mapa de tráfico en vivo ({error_traf_global})."
        )
    elif df_traf_global.empty:
        st.warning(
            "No hay datos de sensores de tráfico disponibles en este momento."
        )
    else:
        st.caption(
            f"Último boletín Informo capturado: "
            f"**{formatear_hora_madrid(ts_traf_global)}** "
            "(refresco automático cada 5 min · lectura simultánea de la ciudad)"
        )

        col_f1, col_f2, col_f3 = st.columns((1.5, 0.8, 1.1))
        with col_f1:
            umbral = st.radio(
                "Nivel de carga vial a mostrar:",
                (
                    "🔴 Congestión y carga alta (carga ≥ 50 %)",
                    "🟣 Tráfico moderado y alto (carga ≥ 25 %)",
                    "🌐 Todos los sensores operativos",
                ),
                horizontal=True,
            )
        with col_f2:
            tipos_disp = ["Todos"] + sorted(
                t for t in df_traf_global["tipo_elem"].dropna().unique() if t
            )
            tipo_sel = st.selectbox("Tipo de vía:", tipos_disp)
        with col_f3:
            modo_vista = st.radio(
                "Modo de visualización:",
                ("🔥 Mapa de calor", "📍 Sensores individuales"),
                horizontal=True,
            )

        df_filtrado = df_traf_global.copy()
        if tipo_sel != "Todos":
            df_filtrado = df_filtrado.loc[
                df_filtrado["tipo_elem"] == tipo_sel
            ].copy()

        t1, t2, t3, t4 = st.columns(4)
        t1.metric(
            "📡 Sensores operativos", f"{len(df_filtrado):,}".replace(",", ".")
        )
        n_cong = int((df_filtrado["carga"] >= 50).sum())
        t2.metric("🔴 Puntos con carga ≥ 50 %", f"{n_cong:,}".replace(",", "."))
        t3.metric("📊 Carga media", f"{df_filtrado['carga'].mean():.1f} %")
        peor_pto = df_filtrado.sort_values("carga", ascending=False).iloc[0]
        t4.metric(
            "⚠ Punto más cargado",
            f"{int(peor_pto['carga'])} %",
            str(peor_pto["descripcion"])[:34],
        )

        if umbral.startswith("🔴"):
            df_mapa = df_filtrado.loc[df_filtrado["carga"] >= 50].copy()
        elif umbral.startswith("🟣"):
            df_mapa = df_filtrado.loc[df_filtrado["carga"] >= 25].copy()
        else:
            df_mapa = df_filtrado

        if df_mapa.empty:
            st.success(
                "Sin sensores por encima de ese umbral de carga ahora mismo."
            )
        else:
            fig_traf = go.Figure()
            if modo_vista.startswith("🔥"):
                fig_traf.add_trace(
                    traza_densidad_mapa(
                        lat=df_mapa["lat"],
                        lon=df_mapa["lon"],
                        z=df_mapa["carga"].clip(0, 100),
                        radius=16,
                        opacity=0.8,
                        colorscale=[
                            [0.0, "rgba(16, 185, 129, 0.0)"],
                            [0.25, "#10B981"],
                            [0.55, "#F59E0B"],
                            [0.85, "#EF4444"],
                            [1.0, "#991B1B"],
                        ],
                        colorbar={"title": "Congestión", "len": 0.85},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            else:
                df_ord = df_mapa.sort_values("carga", ascending=True)
                tamano_pto = 6 if len(df_ord) > 800 else 8
                texto_traf = [
                    f"<b>{r['descripcion'] or r['idelem']}</b> "
                    f"({r['tipo_elem']})<br>"
                    f"Carga: {int(r['carga'])} % · Intensidad: "
                    f"{int(r['intensidad'] or 0)} veh/h<br>"
                    f"Ocupación: {int(r['ocupacion'] or 0)} % · "
                    f"Nivel servicio: {int(r['nivelServicio'] or 0)}"
                    for _, r in df_ord.iterrows()
                ]
                fig_traf.add_trace(
                    traza_mapa(
                        lat=df_ord["lat"],
                        lon=df_ord["lon"],
                        mode="markers",
                        marker={
                            "size": tamano_pto + 2,
                            "color": "#FFFFFF",
                            "opacity": 0.9,
                        },
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                fig_traf.add_trace(
                    traza_mapa(
                        lat=df_ord["lat"],
                        lon=df_ord["lon"],
                        mode="markers",
                        marker={
                            "size": tamano_pto,
                            "opacity": 0.95,
                            "color": df_ord["carga"],
                            "cmin": 0,
                            "cmax": 100,
                            "colorscale": [
                                [0.0, "#10B981"],
                                [0.35, "#F59E0B"],
                                [0.70, "#EF4444"],
                                [1.0, "#991B1B"],
                            ],
                            "showscale": True,
                            "colorbar": {"title": "Carga (%)", "len": 0.85},
                        },
                        text=texto_traf,
                        hoverinfo="text",
                        showlegend=False,
                    )
                )
            configurar_layout_mapa(
                fig_traf,
                float(df_mapa["lat"].mean()),
                float(df_mapa["lon"].mean()),
                zoom=11.7,
            )
            st.plotly_chart(fig_traf, use_container_width=True)

        st.subheader("Top 15 tramos con mayor carga vial en este momento")
        cols_top = [
            "descripcion",
            "tipo_elem",
            "carga",
            "intensidad",
            "ocupacion",
            "nivelServicio",
        ]
        top_traf = (
            df_filtrado.sort_values(
                ["carga", "intensidad"], ascending=[False, False]
            )
            .head(15)
            .loc[:, cols_top]
            .rename(
                columns={
                    "descripcion": "Ubicación",
                    "tipo_elem": "Tipo",
                    "carga": "Carga (%)",
                    "intensidad": "Intensidad (veh/h)",
                    "ocupacion": "Ocupación (%)",
                    "nivelServicio": "Nivel servicio",
                }
            )
        )
        st.dataframe(top_traf, use_container_width=True, hide_index=True)

# --- Pestaña 3: Salud del Sistema y Auditoría GTFS ---
with tab_salud:
    st.subheader("Latido del Sistema Serverless 24/7 (Cloudflare Workers + D1)")
    if error_live:
        st.error(
            f"No se pudo conectar con el servicio en tiempo real ({error_live})."
        )
    else:
        meteo_s = datos_live.get("meteo_actual") or {}
        trafico_s = datos_live.get("trafico_actual") or {}
        cuota_cron = datos_live.get("estado_cuota_emt") or {}
        cuota_web = datos_live.get("estado_cuota_web") or {}
        limites = datos_live.get("limites_emt") or {}

        lim_cron = int(limites.get("cron_diario", 4800))
        lim_web = int(limites.get("web_diario", 150))
        hechas_cron = int(cuota_cron.get("peticiones_realizadas", 0))
        hechas_web = int(cuota_web.get("peticiones_realizadas", 0))

        st.info(
            f"Último pulso del servidor: "
            f"**{formatear_hora_madrid(datos_live.get('generado_utc'))}** "
            "(hora local de Madrid)"
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "📡 Cuota Recolector 24/7",
            f"{hechas_cron:,} / {lim_cron:,}".replace(",", "."),
            f"Parada índice: {cuota_cron.get('indice_parada_actual', 0)}",
        )
        s2.metric(
            "🔎 Cuota Consultas Web",
            f"{hechas_web} / {lim_web}",
            "Cupo protegido (reinicio 00:00 UTC)",
        )
        hora_traf_s = formatear_hora_madrid(
            trafico_s.get("timestamp_captura_utc")
        ).split(" ")[-1]
        kb_traf_s = round(
            (trafico_s.get("tamano_bytes_original") or 0) / 1024, 1
        )
        s3.metric("🚦 Último pulso Tráfico", hora_traf_s, f"{kb_traf_s} KB XML")
        hora_met_s = formatear_hora_madrid(
            meteo_s.get("timestamp_captura_utc")
        ).split(" ")[-1]
        temp_s = meteo_s.get("temperatura_c")
        s4.metric(
            "🌡️ Último pulso AEMET",
            hora_met_s,
            f"{temp_s:.1f} °C" if temp_s is not None else "N/D",
        )

    st.divider()
    st.subheader(
        "Control de Versiones Oficiales del Catálogo GTFS (EMT Madrid)"
    )
    try:
        historial_gtfs, origen_gtfs_hist = cargar_historial_gtfs()
        st.caption(f"Origen de datos: **{origen_gtfs_hist}**")
        if historial_gtfs:
            vigente = historial_gtfs[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "📦 Versión GTFS vigente", vigente.get("version_tag", "N/D")
            )
            c2.metric(
                "📅 Fecha de detección",
                formatear_hora_madrid(vigente.get("detectado_utc")),
            )
            c3.metric(
                "🔍 Última verificación",
                formatear_hora_madrid(
                    vigente.get("ultima_verificacion_utc")
                    or vigente.get("detectado_utc")
                ),
            )

            for item in reversed(historial_gtfs):
                v_tag = item.get("version_tag", "N/D")
                det = formatear_hora_madrid(item.get("detectado_utc"))
                rev = formatear_hora_madrid(
                    item.get("ultima_verificacion_utc") or item.get("detectado_utc")
                )
                tabs = ", ".join(item.get("tablas_modificadas") or ("Inicial",))
                zip_ruta = item.get("ruta_zip", "Inicial")

                with st.expander(f"📦 Versión GTFS: {v_tag} (Detectada: {det})"):
                    col_h1, col_h2 = st.columns(2)
                    col_h1.write(f"**Última revisión:** {rev}")
                    col_h1.write(f"**Tablas modificadas:** {tabs}")
                    col_h2.write(f"**Snapshot ZIP remoto:** `{zip_ruta}`")

                    # Mostrar resumen de cambios (diff) si existe
                    diff = item.get("resumen_cambios")
                    if diff:
                        st.markdown("**📊 Resumen de Cambios (Diff frente a versión previa):**")
                        d1, d2, d3, d4 = st.columns(4)
                        d1.metric("Total Paradas", f"{diff.get('total_paradas', 0):,}".replace(",", "."))
                        d2.metric("Total Líneas", f"{diff.get('total_lineas', 0)}")
                        d3.metric("Delta Viajes", f"{diff.get('viajes_delta', 0):+d}")
                        d4.metric("Delta Frecuencias", f"{diff.get('frecuencias_delta', 0):+d}")

                    # Mostrar cuarentenas auditadas si existen
                    cuar = item.get("cuarentenas")
                    if cuar:
                        st.markdown("**🛡️ Auditoría de Cuarentenas e Incidencias en este ZIP:**")
                        filas_c = [
                            {"Tabla": tbl, "Registros afectados": info.get("cantidad", 0), "Motivo": info.get("motivo", "")}
                            for tbl, info in cuar.items()
                        ]
                        st.dataframe(pd.DataFrame(filas_c), use_container_width=True, hide_index=True)
                    elif v_tag != "20261118_inicial":
                        st.caption("Sin incidencias de cuarentena registradas en esta versión.")
        else:
            st.info(
                "Aún no hay registros en el historial. Aparecerán "
                "automáticamente tras la próxima ejecución programada."
            )
    except Exception as exc:
        st.warning(f"No se pudo cargar el historial ({exc}).")
