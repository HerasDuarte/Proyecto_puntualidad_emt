"""
Módulo de Consolidación Relacional y Feature Engineering Estructural (GTFS EMT Madrid).
Ensambla las 8 tablas saneadas en un único Golden Dataset columnar (maestro_gtfs.parquet)
utilizando el motor de ejecución diferida Polars Lazy para optimizar memoria y velocidad.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def construir_tabla_maestra_gtfs(processed_dir=None, output_path=None):
    """
    Cruza las tablas limpias del catálogo GTFS e inyecta los 9 predictores de
    interacción relacional validados en el análisis exploratorio:
      - hora_entera, tramo_15min
      - tipo_jornada (LABORABLE, VIERNES_VISPERA, FIN_DE_SEMANA)
      - indice_sinuosidad_normalizado (acotado al P95 lineal en rutas circulares)
      - velocidad_teorica_kmh
      - friccion_acumulada_m30
      - hora_seno, hora_coseno
      - saturacion_marquesina_15min

    Retorna el DataFrame consolidado de Polars (2.232.015 filas x 49 columnas base).
    """
    base_dir = Path(processed_dir) if processed_dir else Path("../data/processed")
    ruta_salida = Path(output_path) if output_path else base_dir / "maestro_gtfs.parquet"

    logging.info("Iniciando ensamblaje relacional en Polars Lazy...")

    # 1. Escaneo diferido (Lazy) de las tablas saneadas
    lf_stop_times = pl.scan_csv(
        base_dir / "stop_times_limpio.csv",
        schema_overrides={"trip_id": pl.String, "stop_id": pl.String}
    )

    lf_trips = pl.scan_csv(
        base_dir / "trips_limpio.csv",
        schema_overrides={
            "trip_id": pl.String,
            "route_id": pl.String,
            "service_id": pl.String,
            "block_id": pl.String,
            "shape_id": pl.String,
            "direction_id": pl.Int8
        }
    ).with_columns(
        pl.col("route_id").str.zfill(3)
    )

    lf_routes = pl.scan_csv(
        base_dir / "routes_limpio.csv",
        schema_overrides={"route_id": pl.String, "route_short_name": pl.String}
    ).with_columns(
        pl.col("route_id").str.zfill(3)
    ).select([
        "route_id", "route_short_name", "cabecera_ruta", "destino_ruta", "regimen_horario"
    ])

    lf_stops = pl.scan_csv(
        base_dir / "stops_limpio.csv",
        schema_overrides={"stop_id": pl.String, "stop_code": pl.String}
    ).select([
        "stop_id", "stop_code", "stop_name", "stop_lat", "stop_lon",
        "calle_principal", "en_alta_densidad_real",
        pl.col("Ubicación Territorial").alias("ubicacion_territorial")
    ])

    # Agregación previa de shapes a nivel de shape_id para evitar multiplicación de filas en el JOIN
    lf_shapes_agg = (
        pl.scan_csv(
            base_dir / "shapes_limpio.csv",
            schema_overrides={"shape_id": pl.String}
        )
        .group_by("shape_id")
        .agg([
            pl.col("shape_dist_traveled").max().cast(pl.Int32).alias("longitud_total_ruta_m"),
            pl.col("indice_sinuosidad").first().cast(pl.Float32).alias("indice_sinuosidad"),
            pl.col("puntos_por_km").first().cast(pl.Float32).alias("puntos_por_km")
        ])
    )

    # Agregación previa de frequencies a nivel de trip_id
    lf_freq_agg = (
        pl.scan_csv(
            base_dir / "frequencies_limpio.csv",
            schema_overrides={"trip_id": pl.String}
        )
        .group_by("trip_id")
        .agg([
            pl.col("headway_secs").mean().cast(pl.Float32).alias("headway_medio_secs"),
            pl.col("headway_secs").min().cast(pl.Int32).alias("headway_min_secs"),
            pl.col("autobuses_por_hora").mean().cast(pl.Float32).alias("autobuses_hora_medio"),
            pl.col("autobuses_por_hora").max().cast(pl.Float32).alias("autobuses_hora_max"),
            pl.col("duracion_franja_secs").sum().cast(pl.Int32).alias("cobertura_total_franjas_secs")
        ])
    )

    # 2. Ensamblaje relacional LEFT JOIN preservando el 100% de stop_times
    lf_unido = (
        lf_stop_times
        .join(lf_trips, on="trip_id", how="left")
        .join(lf_routes, on="route_id", how="left")
        .join(lf_stops, on="stop_id", how="left")
        .join(lf_shapes_agg, on="shape_id", how="left")
        .join(lf_freq_agg, on="trip_id", how="left")
    )

    # Materialización intermedia en memoria para calcular cuantiles globales (P95 sinuosidad)
    df_maestro = lf_unido.collect()

    # 3. Feature Engineering Relacional Vectorizado
    es_ruta_circular = (
        (pl.col("cabecera_ruta") == pl.col("destino_ruta")) |
        (pl.col("route_short_name").str.starts_with("C"))
    )

    p95_sinuosidad_lineal = (
        df_maestro
        .filter(~es_ruta_circular)
        .select(pl.col("indice_sinuosidad").quantile(0.95))
        .item()
    )

    df_maestro = df_maestro.with_columns([
        # A) Franjas horarias: hora entera (0-23) y ventana de 15 minutos (0-95)
        (pl.col("hora_llegada_decimal").floor().cast(pl.Int32) % 24).alias("hora_entera"),
        ((pl.col("hora_llegada_decimal") * 4.0).floor().cast(pl.Int32) % 96).alias("tramo_15min"),

        # B) Régimen semanal de demanda comercial (service_id)
        pl.when(pl.col("service_id").str.contains("VV"))
          .then(pl.lit("VIERNES_VISPERA"))
          .when(pl.col("service_id").str.contains("SA|FE"))
          .then(pl.lit("FIN_DE_SEMANA"))
          .otherwise(pl.lit("LABORABLE"))
          .alias("tipo_jornada"),

        # C) Normalización del índice de sinuosidad en líneas circulares reales
        pl.when(es_ruta_circular)
          .then(pl.min_horizontal(pl.col("indice_sinuosidad"), pl.lit(p95_sinuosidad_lineal)))
          .otherwise(pl.col("indice_sinuosidad"))
          .cast(pl.Float32)
          .alias("indice_sinuosidad_normalizado"),

        # D) Velocidad comercial teórica del tramo (km/h)
        pl.when(pl.col("tiempo_tramo_teorico_seg") > 0)
          .then((pl.col("distancia_tramo_m") / pl.col("tiempo_tramo_teorico_seg")) * 3.6)
          .otherwise(0.0)
          .clip(0.0, 80.0)
          .cast(pl.Float32)
          .alias("velocidad_teorica_kmh"),

        # E) Fricción territorial acumulada dentro de la Almendra Central M-30 (escala 0.0-1.0)
        (pl.col("en_alta_densidad_real").cast(pl.Float32) * pl.col("progreso_viaje_pct").cast(pl.Float32))
          .cast(pl.Float32)
          .alias("friccion_acumulada_m30"),

        # F) Codificación trigonométrica cíclica de las 24 horas del día
        (2.0 * np.pi * (pl.col("hora_llegada_decimal") % 24.0) / 24.0).sin().cast(pl.Float32).alias("hora_seno"),
        (2.0 * np.pi * (pl.col("hora_llegada_decimal") % 24.0) / 24.0).cos().cast(pl.Float32).alias("hora_coseno")
    ])

    # G) Saturación simultánea por marquesina, sentido, tipo de jornada y tramo de 15 minutos
    df_maestro = df_maestro.with_columns(
        pl.len()
        .over(["stop_id", "direction_id", "tipo_jornada", "tramo_15min"])
        .cast(pl.Int32)
        .alias("saturacion_marquesina_15min")
    )

    # 4. Exportación columnar comprimida en Parquet ZSTD
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_maestro.write_parquet(ruta_salida, compression="zstd")

    logging.info(
        f"Tabla Maestra consolidada con éxito: {df_maestro.height:,} filas x {df_maestro.width} columnas -> {ruta_salida}"
    )
    return df_maestro


if __name__ == "__main__":
    construir_tabla_maestra_gtfs(processed_dir=Path("data/processed"))
