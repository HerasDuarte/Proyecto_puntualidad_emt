"""
Módulo de Consolidación Relacional y Feature Engineering Estructural (GTFS EMT Madrid).
Ensambla las 8 tablas saneadas en un único Golden Dataset columnar (maestro_gtfs.parquet)
utilizando el motor de ejecución diferida Polars Lazy para optimizar memoria y velocidad.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def construir_tabla_maestra_gtfs(processed_dir=None, output_path=None):
    """
    Cruza las tablas limpias del catálogo GTFS e inyecta predictores estructurales y
    de interacción relacional validados:
      - hora_entera, hora_seno, hora_coseno
      - tipo_jornada (LABORABLE, VIERNES_VISPERA, FIN_DE_SEMANA)
      - es_linea_circular, tipo_servicio_emt, distancia_centro_km0_m
      - hub_conectividad_parada, es_cabecera_terminal
      - velocidad_teorica_kmh, friccion_acumulada_m30, saturacion_marquesina_15min

    Retorna el DataFrame consolidado de Polars (47 columnas optimizadas).
    """
    base_dir = Path(processed_dir) if processed_dir else Path("../data/processed")
    ruta_salida = (
        Path(output_path) if output_path else base_dir / "maestro_gtfs.parquet"
    )

    logging.info("Iniciando ensamblaje relacional en Polars Lazy...")

    # 1. Escaneo diferido (Lazy) de las tablas saneadas
    lf_stop_times = pl.scan_csv(
        base_dir / "stop_times_limpio.csv",
        schema_overrides={"trip_id": pl.String, "stop_id": pl.String},
    )

    lf_trips = pl.scan_csv(
        base_dir / "trips_limpio.csv",
        schema_overrides={
            "trip_id": pl.String,
            "route_id": pl.String,
            "service_id": pl.String,
            "block_id": pl.String,
            "shape_id": pl.String,
            "direction_id": pl.Int8,
        },
    ).with_columns(pl.col("route_id").str.zfill(3))

    lf_routes = (
        pl.scan_csv(
            base_dir / "routes_limpio.csv",
            schema_overrides={
                "route_id": pl.String,
                "route_short_name": pl.String,
                "tipo_servicio_emt": pl.String,
            },
        )
        .with_columns(
            [
                pl.col("route_id").str.zfill(3),
                pl.col("es_linea_circular").cast(pl.Boolean),
            ]
        )
        .select(
            [
                "route_id",
                "route_short_name",
                "cabecera_ruta",
                "destino_ruta",
                "regimen_horario",
                "es_linea_circular",
                "tipo_servicio_emt",
            ]
        )
    )

    lf_stops = pl.scan_csv(
        base_dir / "stops_limpio.csv",
        schema_overrides={"stop_id": pl.String, "stop_code": pl.String},
    ).select(
        [
            "stop_id",
            "stop_code",
            "stop_name",
            "stop_lat",
            "stop_lon",
            "calle_principal",
            pl.col("distancia_centro_km0_m").cast(pl.Float32),
            "en_alta_densidad_real",
            pl.col("Ubicación Territorial").alias("ubicacion_territorial"),
        ]
    )

    # Agregación previa de shapes a nivel de shape_id
    lf_shapes_agg = (
        pl.scan_csv(
            base_dir / "shapes_limpio.csv",
            schema_overrides={"shape_id": pl.String},
        )
        .group_by("shape_id")
        .agg(
            [
                pl.col("shape_dist_traveled")
                .max()
                .cast(pl.Int32)
                .alias("longitud_total_ruta_m"),
                pl.col("indice_sinuosidad")
                .first()
                .cast(pl.Float32)
                .alias("indice_sinuosidad"),
                pl.col("puntos_por_km")
                .first()
                .cast(pl.Float32)
                .alias("puntos_por_km"),
            ]
        )
    )

    # Agregación previa de frequencies a nivel de trip_id (sin métricas redundantes)
    lf_freq_agg = (
        pl.scan_csv(
            base_dir / "frequencies_limpio.csv",
            schema_overrides={"trip_id": pl.String},
        )
        .group_by("trip_id")
        .agg(
            [
                pl.col("headway_secs")
                .mean()
                .cast(pl.Float32)
                .alias("headway_medio_secs"),
                pl.col("autobuses_por_hora")
                .mean()
                .cast(pl.Float32)
                .alias("autobuses_hora_medio"),
                pl.col("duracion_franja_secs")
                .sum()
                .cast(pl.Int32)
                .alias("cobertura_total_franjas_secs"),
            ]
        )
    )

    # 2. Ensamblaje relacional LEFT JOIN preservando el 100% de stop_times
    lf_unido = (
        lf_stop_times.join(lf_trips, on="trip_id", how="left")
        .join(lf_routes, on="route_id", how="left")
        .join(lf_stops, on="stop_id", how="left")
        .join(lf_shapes_agg, on="shape_id", how="left")
        .join(lf_freq_agg, on="trip_id", how="left")
    )

    df_maestro = lf_unido.collect()

    # 3. Feature Engineering Relacional Vectorizado
    df_maestro = df_maestro.with_columns(
        [
            (pl.col("hora_llegada_decimal").floor().cast(pl.Int32) % 24).alias(
                "hora_entera"
            ),
            ((pl.col("hora_llegada_decimal") * 4.0).floor().cast(pl.Int32) % 96).alias(
                "_tramo_15min_tmp"
            ),
            pl.when(pl.col("service_id").str.contains("VV"))
            .then(pl.lit("VIERNES_VISPERA"))
            .when(pl.col("service_id").str.contains("SA|FE"))
            .then(pl.lit("FIN_DE_SEMANA"))
            .otherwise(pl.lit("LABORABLE"))
            .alias("tipo_jornada"),
            pl.when(pl.col("tiempo_tramo_teorico_seg") > 0)
            .then(
                (pl.col("distancia_tramo_m") / pl.col("tiempo_tramo_teorico_seg")) * 3.6
            )
            .otherwise(0.0)
            .clip(0.0, 80.0)
            .cast(pl.Float32)
            .alias("velocidad_teorica_kmh"),
            (
                pl.col("en_alta_densidad_real").cast(pl.Float32)
                * pl.col("progreso_viaje_pct").cast(pl.Float32)
            )
            .cast(pl.Float32)
            .alias("friccion_acumulada_m30"),
            (2.0 * np.pi * (pl.col("hora_llegada_decimal") % 24.0) / 24.0)
            .sin()
            .cast(pl.Float32)
            .alias("hora_seno"),
            (2.0 * np.pi * (pl.col("hora_llegada_decimal") % 24.0) / 24.0)
            .cos()
            .cast(pl.Float32)
            .alias("hora_coseno"),
        ]
    )

    # Variables de interacción de red (marquesina, conectividad y cabecera/terminal)
    df_maestro = df_maestro.with_columns(
        [
            pl.len()
            .over(["stop_id", "direction_id", "tipo_jornada", "_tramo_15min_tmp"])
            .cast(pl.Int32)
            .alias("saturacion_marquesina_15min"),
            pl.col("route_id")
            .n_unique()
            .over("stop_id")
            .cast(pl.Int16)
            .alias("hub_conectividad_parada"),
            (
                (pl.col("stop_sequence") == 1)
                | (
                    pl.col("stop_sequence")
                    == pl.col("stop_sequence").max().over("trip_id")
                )
            ).alias("es_cabecera_terminal"),
        ]
    ).drop("_tramo_15min_tmp")

    # 4. Exportación columnar comprimida en Parquet ZSTD
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_maestro.write_parquet(ruta_salida, compression="zstd")

    logging.info(
        f"Tabla Maestra consolidada con éxito: {df_maestro.height:,} filas x "
        f"{df_maestro.width} columnas -> {ruta_salida}"
    )
    return df_maestro


if __name__ == "__main__":
    construir_tabla_maestra_gtfs(processed_dir=Path("data/processed"))
