import pandas as pd
import sys
from db_connection import connect_to_mongo, insert_data_in_chunks

def load_data(csv_path):

        # Configurar la codificación de salida para la consola
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    """
    Carga los datos desde un archivo CSV.
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print("Datos cargados exitosamente desde el CSV.")
        return df
    except Exception as e:
        print(f"Error al cargar el CSV: {e}")
        return None

def upload_data_to_mongo(df, collection_name):
    """
    Sube un DataFrame a una colección en MongoDB.
    """
    client = connect_to_mongo()
    if not client:
        print("No se pudo establecer conexión con MongoDB.")
        return

    # Llamar a la función de inserción en chunks
    insert_data_in_chunks(df, collection_name, client)
