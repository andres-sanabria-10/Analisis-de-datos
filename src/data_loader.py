import pandas as pd
import sys

def load_data(victimas_path):
    # Configurar la codificación de salida para la consola
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    try:
        df_victimas = pd.read_csv(victimas_path, encoding='utf-8')

        print("Datos cargados exitosamente con codificación: utf-8")
        
        return df_victimas
    except UnicodeDecodeError:
        print("Error: No se pudo decodificar el archivo con codificación utf-8.")
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
    
    raise ValueError("No se pudo cargar los datos.")
