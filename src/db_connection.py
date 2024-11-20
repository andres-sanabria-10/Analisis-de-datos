import pandas as pd
from pymongo import MongoClient
import os

def connect_to_mongo():
    """
    Conecta a MongoDB utilizando la URI configurada en las variables de entorno.
    """
    uri = os.getenv("DB_URI")
    try:
        client = MongoClient(uri, retryWrites=True, w="majority", serverSelectionTimeoutMS=30000)
        print("Conexión exitosa a MongoDB Atlas.")
        return client
    except Exception as e:
        print(f"Error al conectar a MongoDB: {e}")
        return None

def insert_data_in_chunks(df, collection_name, client):
    """
    Inserta los datos del DataFrame en MongoDB en trozos pequeños.
    """
    try:
        # Seleccionar la base de datos y colección
        db = client['delitos_sexuales']  # Nombre de la base de datos
        collection = db[collection_name]  # Nombre de la colección

        # Vaciar la colección antes de insertar nuevos datos
        collection.delete_many({})
        print(f"Todos los documentos en la colección '{collection_name}' han sido eliminados.")

        # Dividir los datos en chunks de 5000 registros (ajusta este valor según sea necesario)
        chunk_size = 5000
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            data = chunk.to_dict("records")  # Convertir el chunk a lista de diccionarios
            collection.insert_many(data, ordered=False)
            print(f"{len(data)} registros insertados en el chunk.")
        
        print("Datos insertados exitosamente en MongoDB.")
    except Exception as e:
        print(f"Error al insertar datos en MongoDB: {e}")
