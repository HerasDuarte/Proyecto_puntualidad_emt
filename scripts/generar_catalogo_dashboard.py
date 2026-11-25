"""
Genera el catálogo GTFS ligero en Parquet para el dashboard interactivo.
Extrae paradas, líneas, recorridos con tiempos acumulados, trazados y
frecuencias oficiales por franja horaria desde data/raw/gtfs/.
"""

import argparse
import logging
import sys
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import DIR_GTFS_ACTIVO, DIR_PROCESSED  # noqa: E402
from src.utils import obtener_cliente_hf  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DIR_CATALOGO_DASHBOARD = DIR_PROCESSED / "catalogo_dashboard"
RUTA_CATALOGO_HF = "catalogo_dashboard"


def _expr_hora_a_segundos(col: str) -> pl.Expr:
    """Convierte cadenas HH:MM:SS (incluidas >24h o 1 dígito) a segundos."""
    partes = pl.col(col).str.strip_chars().str.split_exact(":", 2)
    return (
        partes.struct.field("field_0").cast(pl.Int32) * 3600
        + partes.struct.field("field_1").cast(pl.Int32) * 60
        + partes.struct.field("field_2").cast(pl.Int32)
    )


def construir_catalogo_dashboard(
    dir_gtfs: Path = DIR_GTFS_ACTIVO,
    dir_salida: Path = DIR_CATALOGO_DASHBOARD,
) -> dict[str, Path]:
    """Construye los 5 archivos Parquet ligeros a partir del GTFS activo."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    logging.info("Leyendo tablas GTFS desde %s...", dir_gtfs)

    df_routes = pl.read_csv(
        dir_gtfs / "routes.txt",
        schema_overrides={
            "route_id": pl.String,
            "route_short_name": pl.String,
            "route_long_name": pl.String,
            "route_color": pl.String,
        },
    ).select(
        [
            pl.col("route_id").str.strip_chars(),
            pl.col("route_short_name").str.strip_chars().alias("line"),
            pl.col("route_long_name").str.strip_chars(),
            pl.col("route_color").fill_null("0178BC").str.strip_chars(),
        ]
    )

    df_stops = pl.read_csv(
        dir_gtfs / "stops.txt",
        schema_overrides={"stop_id": pl.String, "stop_name": pl.String},
    ).select(
        [
            pl.col("stop_id").str.strip_chars(),
            pl.col("stop_name").str.strip_chars(),
            pl.col("stop_lat").cast(pl.Float32),
            pl.col("stop_lon").cast(pl.Float32),
        ]
    )

    df_calendar = pl.read_csv(
        dir_gtfs / "calendar.txt",
        schema_overrides={"service_id": pl.String},
    ).with_columns(
        pl.when(pl.col("saturday") == 1)
        .then(pl.lit("SA"))
        .when(pl.col("sunday") == 1)
        .then(pl.lit("FE"))
        .otherwise(pl.lit("LA"))
        .alias("tipo_dia")
    ).select([pl.col("service_id").str.strip_chars(), "tipo_dia"])

    df_trips = (
        pl.read_csv(
            dir_gtfs / "trips.txt",
            schema_overrides={
                "route_id": pl.String,
                "service_id": pl.String,
                "trip_id": pl.String,
                "trip_headsign": pl.String,
                "shape_id": pl.String,
            },
        )
        .select(
            [
                pl.col("route_id").str.strip_chars(),
                pl.col("service_id").str.strip_chars(),
                pl.col("trip_id").str.strip_chars(),
                pl.col("trip_headsign").fill_null("").str.strip_chars(),
                pl.col("direction_id").cast(pl.Int8),
                pl.col("shape_id").str.strip_chars(),
            ]
        )
        .join(df_routes, on="route_id", how="inner")
        .join(df_calendar, on="service_id", how="left")
        .with_columns(pl.col("tipo_dia").fill_null("LA"))
    )

    df_stop_times = pl.scan_csv(
        dir_gtfs / "stop_times.txt",
        schema_overrides={
            "trip_id": pl.String,
            "arrival_time": pl.String,
            "stop_id": pl.String,
        },
    ).select(
        [
            pl.col("trip_id").str.strip_chars(),
            _expr_hora_a_segundos("arrival_time").alias("arrival_sec"),
            pl.col("stop_id").str.strip_chars(),
            pl.col("stop_sequence").cast(pl.Int16),
            pl.col("shape_dist_traveled")
            .fill_null(0)
            .cast(pl.Int32)
            .alias("dist_m"),
        ]
    )

    # 1. Paradas + líneas que pasan por cada parada
    lineas_por_parada = (
        df_stop_times.join(
            df_trips.select(["trip_id", "line"]).lazy(),
            on="trip_id",
            how="inner",
        )
        .select(["stop_id", "line"])
        .unique()
        .collect()
        .sort(["stop_id", "line"])
        .group_by("stop_id")
        .agg(pl.col("line").str.join(", ").alias("lineas"))
    )
    df_paradas_out = df_stops.join(lineas_por_parada, on="stop_id", how="left").with_columns(
        pl.col("lineas").fill_null("")
    )

    # 2. Viaje representativo por (line, direction_id): más paradas, prioridad LA
    resumen_viajes = (
        df_stop_times.group_by("trip_id")
        .agg(pl.len().alias("n_paradas"))
        .collect()
        .join(df_trips, on="trip_id", how="inner")
        .with_columns(
            pl.when(pl.col("tipo_dia") == "LA")
            .then(1)
            .otherwise(0)
            .alias("prio_la")
        )
        .sort(["line", "direction_id", "prio_la", "n_paradas"], descending=[False, False, True, True])
        .unique(subset=["line", "direction_id"], keep="first")
    )

    df_lineas_out = resumen_viajes.select(
        [
            "line",
            "route_id",
            "route_long_name",
            "route_color",
            "direction_id",
            pl.col("trip_headsign").alias("headsign"),
            "shape_id",
            "trip_id",
        ]
    ).sort(["line", "direction_id"])

    # 3. Secuencia de paradas y minutos acumulados desde cabecera
    df_recorridos_out = (
        df_stop_times.join(
            df_lineas_out.select(["trip_id", "line", "direction_id"]).lazy(),
            on="trip_id",
            how="inner",
        )
        .collect()
        .sort(["line", "direction_id", "stop_sequence"])
        .with_columns(
            (
                (pl.col("arrival_sec") - pl.col("arrival_sec").first().over(["line", "direction_id"]))
                / 60.0
            )
            .round(1)
            .cast(pl.Float32)
            .alias("min_desde_cabecera")
        )
        .join(
            df_stops.select(["stop_id", "stop_name", "stop_lat", "stop_lon"]),
            on="stop_id",
            how="left",
        )
        .select(
            [
                "line",
                "direction_id",
                "stop_sequence",
                "stop_id",
                "stop_name",
                "stop_lat",
                "stop_lon",
                "min_desde_cabecera",
                "dist_m",
            ]
        )
    )

    # 4. Trazados de los shape_id representativos
    shapes_necesarios = df_lineas_out.select(["shape_id"]).unique()
    df_trazados_out = (
        pl.scan_csv(
            dir_gtfs / "shapes.txt",
            schema_overrides={"shape_id": pl.String},
        )
        .select(
            [
                pl.col("shape_id").str.strip_chars(),
                pl.col("shape_pt_sequence").cast(pl.Int32).alias("seq"),
                pl.col("shape_pt_lat").cast(pl.Float32).alias("lat"),
                pl.col("shape_pt_lon").cast(pl.Float32).alias("lon"),
            ]
        )
        .join(shapes_necesarios.lazy(), on="shape_id", how="inner")
        .collect()
        .sort(["shape_id", "seq"])
    )

    # 5. Frecuencias oficiales por (line, direction_id, tipo_dia, hora)
    df_freq_raw = pl.read_csv(
        dir_gtfs / "frequencies.txt",
        schema_overrides={
            "trip_id": pl.String,
            "start_time": pl.String,
            "end_time": pl.String,
        },
    ).with_columns(
        [
            pl.col("trip_id").str.strip_chars(),
            (_expr_hora_a_segundos("start_time") // 3600).alias("h_ini"),
            ((_expr_hora_a_segundos("end_time") - 1).clip(lower_bound=0) // 3600).alias("h_fin"),
        ]
    ).with_columns(
        pl.when(pl.col("h_fin") < pl.col("h_ini"))
        .then(pl.col("h_ini"))
        .otherwise(pl.col("h_fin"))
        .alias("h_fin")
    )

    df_frecuencias_out = (
        df_freq_raw.join(
            df_trips.select(["trip_id", "line", "direction_id", "tipo_dia"]),
            on="trip_id",
            how="inner",
        )
        .with_columns(pl.int_ranges("h_ini", pl.col("h_fin") + 1).alias("hora_rango"))
        .explode("hora_rango")
        .with_columns((pl.col("hora_rango") % 24).cast(pl.Int8).alias("hora"))
        .group_by(["line", "direction_id", "tipo_dia", "hora"])
        .agg((pl.col("headway_secs").median() / 60.0).round(1).cast(pl.Float32).alias("headway_min"))
        .sort(["line", "direction_id", "tipo_dia", "hora"])
    )

    archivos = {
        "paradas": dir_salida / "paradas.parquet",
        "lineas": dir_salida / "lineas.parquet",
        "recorridos_paradas": dir_salida / "recorridos_paradas.parquet",
        "trazados": dir_salida / "trazados.parquet",
        "frecuencias_oficiales": dir_salida / "frecuencias_oficiales.parquet",
    }
    df_paradas_out.write_parquet(archivos["paradas"], compression="zstd")
    df_lineas_out.drop("trip_id").write_parquet(archivos["lineas"], compression="zstd")
    df_recorridos_out.write_parquet(archivos["recorridos_paradas"], compression="zstd")
    df_trazados_out.write_parquet(archivos["trazados"], compression="zstd")
    df_frecuencias_out.write_parquet(archivos["frecuencias_oficiales"], compression="zstd")

    total_kb = sum(p.stat().st_size for p in archivos.values()) / 1024
    logging.info("Catálogo generado en %s (peso total: %.1f KB).", dir_salida, total_kb)
    for nombre, ruta in archivos.items():
        logging.info(" - %s: %.1f KB", nombre, ruta.stat().st_size / 1024)
    return archivos


def subir_catalogo_hf(archivos: dict[str, Path]) -> None:
    """Sube los archivos del catálogo ligero al dataset de Hugging Face."""
    load_dotenv()
    api, dataset_id = obtener_cliente_hf()
    for ruta in archivos.values():
        api.upload_file(
            path_or_fileobj=str(ruta),
            path_in_repo=f"{RUTA_CATALOGO_HF}/{ruta.name}",
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=f"chore(dashboard): actualiza catálogo ligero {ruta.name}",
        )
    logging.info("Catálogo ligero sincronizado en Hugging Face (%s/%s).", dataset_id, RUTA_CATALOGO_HF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador del catálogo ligero GTFS.")
    parser.add_argument(
        "--subir-hf",
        action="store_true",
        help="Sube además el catálogo ligero generado a Hugging Face Hub.",
    )
    args = parser.parse_args()
    salida = construir_catalogo_dashboard()
    if args.subir_hf:
        subir_catalogo_hf(salida)
