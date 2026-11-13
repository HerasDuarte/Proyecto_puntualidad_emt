import polars as pl
import logging
from pathlib import Path


def construir_tabla_maestra_gtfs(processed_dir=None):
    """
    Consolida las 6 tablas operativas saneadas del estándar GTFS en una única
    Tabla Analítica Maestra (Golden Dataset) mediante procesamiento multihilo en Polars,
    garantizando la preservación íntegra del 100% de los registros de stop_times_limpio.csv.
    """
    base_path = Path(processed_dir) if processed_dir else Path('../data/processed')

    logging.info("Iniciando consolidación relacional de la Tabla Maestra GTFS en Polars...")

    # 1. Escaneo Lazy de la matriz transaccional base (stop_times)
    lf_stop_times = pl.scan_csv(
        base_path / 'stop_times_limpio.csv',
        schema_overrides={
            'trip_id': pl.String,
            'arrival_time': pl.String,
            'departure_time': pl.String,
            'stop_id': pl.String,
            'stop_sequence': pl.Int16,
            'shape_dist_traveled': pl.Float32,
            'tiempo_tramo_teorico_seg': pl.Int64,
            'distancia_tramo_m': pl.Float32,
            'hora_llegada_decimal': pl.Float64,
            'progreso_viaje_pct': pl.Float32,
            'densidad_paradas_acumuladas': pl.Float32
        }
    )

    # 2. Escaneo Lazy de viajes (trips)
    lf_trips = pl.scan_csv(
        base_path / 'trips_limpio.csv',
        schema_overrides={
            'route_id': pl.String,
            'service_id': pl.String,
            'trip_id': pl.String,
            'trip_headsign': pl.String,
            'direction_id': pl.Int8,
            'block_id': pl.String,
            'shape_id': pl.String,
            'volumen_viajes_bloque': pl.Int16
        }
    )

    # 3. Escaneo Lazy de líneas comerciales (routes)
    lf_routes = pl.scan_csv(
        base_path / 'routes_limpio.csv',
        schema_overrides={
            'route_id': pl.String,
            'route_short_name': pl.String,
            'route_long_name': pl.String,
            'cabecera_ruta': pl.String,
            'destino_ruta': pl.String,
            'regimen_horario': pl.String
        }
    ).select([
        'route_id', 'route_short_name', 'cabecera_ruta', 'destino_ruta', 'regimen_horario'
    ])

    # 4. Escaneo Lazy de paradas físicas (stops)
    lf_stops = pl.scan_csv(
        base_path / 'stops_limpio.csv',
        schema_overrides={
            'stop_id': pl.String,
            'stop_code': pl.String,
            'stop_name': pl.String,
            'stop_lat': pl.Float32,
            'stop_lon': pl.Float32,
            'calle_principal': pl.String,
            'en_alta_densidad_real': pl.Boolean,
            'Ubicación Territorial': pl.String
        }
    ).rename({'Ubicación Territorial': 'ubicacion_territorial'})

    # 5. Agregación geométrica única por trazado (shapes)
    lf_shapes_agg = (
        pl.scan_csv(
            base_path / 'shapes_limpio.csv',
            schema_overrides={
                'shape_id': pl.String,
                'shape_dist_traveled': pl.Int32,
                'indice_sinuosidad': pl.Float32,
                'puntos_por_km': pl.Float32
            }
        )
        .group_by('shape_id')
        .agg([
            pl.col('shape_dist_traveled').max().alias('longitud_total_ruta_m'),
            pl.col('indice_sinuosidad').first().alias('indice_sinuosidad'),
            pl.col('puntos_por_km').first().alias('puntos_por_km')
        ])
    )

    # 6. Agregación única por expedición (frequencies) para preservar el 100% de filas
    lf_frequencies_agg = (
        pl.scan_csv(
            base_path / 'frequencies_limpio.csv',
            schema_overrides={
                'trip_id': pl.String,
                'headway_secs': pl.Int32,
                'autobuses_por_hora': pl.Float32,
                'duracion_franja_secs': pl.Int32
            }
        )
        .group_by('trip_id')
        .agg([
            pl.col('headway_secs').mean().round(1).cast(pl.Float32).alias('headway_medio_secs'),
            pl.col('headway_secs').min().alias('headway_min_secs'),
            pl.col('autobuses_por_hora').mean().round(2).cast(pl.Float32).alias('autobuses_hora_medio'),
            pl.col('autobuses_por_hora').max().round(2).cast(pl.Float32).alias('autobuses_hora_max'),
            pl.col('duracion_franja_secs').sum().alias('cobertura_total_franjas_secs')
        ])
    )

    # 7. Ensamblaje relacional 1-a-1 mediante Left Joins controlados
    lf_maestro = (
        lf_stop_times
        .join(lf_trips, on='trip_id', how='left')
        .join(lf_routes, on='route_id', how='left')
        .join(lf_stops, on='stop_id', how='left')
        .join(lf_shapes_agg, on='shape_id', how='left')
        .join(lf_frequencies_agg, on='trip_id', how='left')
        .sort(['trip_id', 'stop_sequence'])
    )

    # 8. Validación dinámica de cardinalidad frente a stop_times_limpio.csv
    filas_esperadas = lf_stop_times.select(pl.len()).collect().item()
    df_maestro = lf_maestro.collect()

    assert df_maestro.height > 0, "Error crítico: La Tabla Maestra GTFS está vacía."
    assert df_maestro.height == filas_esperadas, (
        f"Error de cardinalidad: Se esperaban {filas_esperadas:,} filas (stop_times_limpio) "
        f"y se obtuvieron {df_maestro.height:,}"
    )

    logging.info(
        f"Tabla Maestra GTFS consolidada con éxito: {df_maestro.height:,} filas x {df_maestro.width} columnas."
    )
    return df_maestro
