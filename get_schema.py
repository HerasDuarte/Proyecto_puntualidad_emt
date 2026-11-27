import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def get_csv_schema(file_path):
    """Obtiene el esquema de un archivo CSV"""
    try:
        # Leer solo las primeras filas para obtener el esquema
        df = pd.read_csv(file_path, nrows=5)
        schema_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            schema_info.append(f"  {col}: {dtype}")
        return schema_info
    except Exception as e:
        return [f"Error al leer CSV: {str(e)}"]

def get_parquet_schema(file_path):
    """Obtiene el esquema de un archivo Parquet"""
    try:
        # Leer el esquema del archivo parquet
        table = pq.read_table(file_path)
        schema_info = []
        for field in table.schema:
            schema_info.append(f"  {field.name}: {field.type}")
        return schema_info
    except Exception as e:
        return [f"Error al leer Parquet: {str(e)}"]

def main():
    path = 'c:/Users/nemra/Desktop/Proyecto_puntualidad_emt/data/processed'
    
    # Obtener todos los archivos CSV y Parquet (excluyendo quarantine)
    files = [f for f in os.listdir(path) if (f.endswith('.csv') or f.endswith('.parquet')) and 'quarantine' not in f]
    
    print("=== ESQUEMA DE ARCHIVOS EN data/processed ===")
    
    for file in files:
        file_path = os.path.join(path, file)
        print(f"\n--- {file} ---")
        
        if file.endswith('.csv'):
            schema_info = get_csv_schema(file_path)
        elif file.endswith('.parquet'):
            schema_info = get_parquet_schema(file_path)
        else:
            continue
            
        for info in schema_info:
            print(info)

if __name__ == "__main__":
    main()