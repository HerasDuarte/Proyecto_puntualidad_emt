import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.consolidacion import construir_tabla_maestra_gtfs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    processed_dir = ROOT_DIR / "data" / "processed"
    output_parquet = processed_dir / "maestro_gtfs.parquet"

    print(
        "====== CONSTRUCCIÓN DE LA TABLA ANALÍTICA MAESTRA "
        "(MAESTRO_GTFS.PARQUET) ======\n"
    )

    df_maestro = construir_tabla_maestra_gtfs(
        processed_dir=processed_dir,
        output_path=output_parquet,
    )

    peso_mb = output_parquet.stat().st_size / (1024 * 1024)
    print(f"\n[ÉXITO] Archivo maestro generado en: {output_parquet}")
    print(f"-> Dimensiones finales: {df_maestro.height:,} filas x {df_maestro.width} columnas")
    print(f"-> Tamaño en disco (Parquet ZSTD): {peso_mb:.2f} MB")
    print(f"-> Columnas consolidadas ({df_maestro.width}):")
    print(list(df_maestro.columns))


if __name__ == "__main__":
    main()
