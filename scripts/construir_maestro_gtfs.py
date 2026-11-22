import logging
from pathlib import Path

# Añadir la raíz del proyecto al PYTHONPATH para importar src
from src.consolidacion import construir_tabla_maestra_gtfs

ROOT_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    processed_dir = ROOT_DIR / 'data' / 'processed'
    output_parquet = processed_dir / 'maestro_gtfs.parquet'

    print("====== CONSTRUCCIÓN DE LA TABLA ANALÍTICA MAESTRA (MAESTRO_GTFS.PARQUET) ======\n")

    df_maestro = construir_tabla_maestra_gtfs(processed_dir=processed_dir)

    # Persistencia en formato binario columna Parquet (compresión zstd de alto rendimiento)
    df_maestro.write_parquet(output_parquet, compression='zstd')

    peso_mb = output_parquet.stat().st_size / (1024 * 1024)
    print(f"\n[ÉXITO] Archivo maestro generado en: {output_parquet}")
    print(f"-> Dimensiones finales: {df_maestro.height:,} filas x {df_maestro.width} columnas")
    print(f"-> Tamaño en disco (Parquet ZSTD): {peso_mb:.2f} MB")
    print(f"-> Columnas consolidadas ({df_maestro.width}):")
    print(list(df_maestro.columns))


if __name__ == '__main__':
    main()
