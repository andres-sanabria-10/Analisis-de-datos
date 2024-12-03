import pandas as pd
from tabulate import tabulate

def clean_and_prepare_data(df):
    """
    Limpia y transforma los datos para su análisis y almacenamiento.
    """
    try:
        print("Iniciando la limpieza y preparación de datos...")

        # Reemplazar campos vacíos con "SIN DATO"
        df.fillna("SIN DATO", inplace=True)

        # Convertir las columnas de año a tipo numérico
        year_columns = ['AÑO_HECHOS', 'AÑO_ENTRADA', 'AÑO_DENUNCIA']
        for column in year_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')  # Convierte valores no numéricos a NaN

        # Mostrar un resumen de los datos después de la limpieza
        print("Resumen después de la limpieza:")
        print(df.info())
        print("Primeras filas del DataFrame después de la limpieza:")
        print(df.head())

        return df
    except Exception as e:
        print(f"Error durante la preparación de datos: {e}")
        return None

def explore_data(df):
    """
    Realiza un entendimiento básico de los datos, mostrando información clave.
    """
    try:
        print("Exploración de los datos:")

        # Información general del DataFrame
        print("\nInformación general:")
        print(df.info())

        # Resumen estadístico de las columnas numéricas
        print("\nResumen estadístico de columnas numéricas:")
        print(df.describe())

        # Ver valores únicos en cada columna
        print("\nValores únicos por columna:")
        for column in df.columns:
            print(f"- {column}: {df[column].nunique()} valores únicos")

        return True
    except Exception as e:
        print(f"Error durante la exploración de datos: {e}")
        return False



def eliminar_columnas(df, columnas_a_eliminar):
    # Verificar que las columnas a eliminar existen en el DataFrame
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    
    # Eliminar las columnas
    df_clean = df.drop(columns=columnas_existentes)
    
    # Obtener los nombres de las columnas restantes
    nombres_columnas_restantes = df_clean.columns.tolist()
    
    # Imprimir los nombres de las columnas restantes en forma de tabla
    print("Nombres de las columnas restantes:")
    print(tabulate([[col] for col in nombres_columnas_restantes], headers=["Columnas"], tablefmt="grid"))
    
    return df_clean  # Asegúrate de devolver el nuevo DataFrame

def combinar_columnas(df, col1, col2, nueva_columna):
    # Crear una copia del DataFrame para evitar modificaciones inesperadas
    df_combined = df.copy()
    
    # Combinar las dos columnas
    df_combined[nueva_columna] = df_combined[col1] + ', ' + df_combined[col2]
    
    # Imprimir la nueva columna creada
    print(f"Nueva columna '{nueva_columna}':")
    print(df_combined[nueva_columna].head())
    
    return df_combined  # Devolver el nuevo DataFrame

def limpiar_columna_delito(df, columna='DELITO'):

    try:
        if columna in df.columns:
            df[columna] = df[columna].str.split('ART.').str[0].str.strip()
            print(f"Columna '{columna}' procesada correctamente.")
        else:
            print(f"La columna '{columna}' no existe en el DataFrame.")
        return df
    except Exception as e:
        print(f"Error al procesar la columna '{columna}': {e}")
        return df


def cambiar_nombre_columna(df, columna_antigua, columna_nueva):
    try:
        if columna_antigua in df.columns:
            df = df.rename(columns={columna_antigua: columna_nueva})
            print(f"Columna '{columna_antigua}' renombrada a '{columna_nueva}' correctamente.")
        else:
            print(f"La columna '{columna_antigua}' no existe en el DataFrame.")
        return df
    except Exception as e:
        print(f"Error al renombrar la columna: {e}")
        return df
    
    
def eliminar_registros(df):
   
    try:
        print("Eliminando registros con valores 'SIN DATO' en las columnas específicas...")
        
        # Condiciones para eliminar registros
        condiciones = (
            (df['ETAPA_CASO'] == "SIN DATO") |
            (df['PAÍS_HECHO'] == "SIN DATO") |
            (df['DEPARTAMENTO_HECHO'] == "SIN DATO") |
            (df['MUNICIPIO_HECHO'] == "SIN DATO") |
            (df['AÑO_DENUNCIA'] == "SIN DATO")
        )
        
        # Contar registros antes de eliminar
        registros_antes = len(df)
        
        # Filtrar registros que no cumplen las condiciones
        df_filtrado = df[~condiciones]
        
        # Contar registros después de eliminar
        registros_despues = len(df_filtrado)
        
        print(f"Registros eliminados: {registros_antes - registros_despues}")
        print(f"Registros restantes: {registros_despues}")
        
        return df_filtrado
    except Exception as e:
        print(f"Error al eliminar registros: {e}")
        return df

def reemplazar_sin_dato_sexo(df, columna='SEXO'):

    try:
        print(f"Reemplazando 'SIN DATO' por 'DESCONOCIDO' en la columna '{columna}'...")
        antes_reemplazo = df[columna].str.contains("SIN DATO").sum()      
        df[columna] = df[columna].replace("SIN DATO", "DESCONOCIDO")
        despues_reemplazo = df[columna].str.contains("SIN DATO").sum()
        reemplazos_hechos = antes_reemplazo - despues_reemplazo
        print(f"Se han reemplazado {reemplazos_hechos} valores 'SIN DATO' por 'DESCONOCIDO' en la columna '{columna}'.")
        return df
    except Exception as e:
        print(f"Error al reemplazar los valores en la columna '{columna}': {e}")
        return df

def reemplazar_sin_dato_pais_nacimiento(df, columna='PAÍS_NACIMIENTO_VICTIMA'):
    try:
        print(f"Reemplazando 'SIN DATO' por 'DESCONOCIDO' en la columna '{columna}'...")
        antes_reemplazo = df[columna].str.contains("SIN DATO").sum()            
        df[columna] = df[columna].replace("SIN DATO", "DESCONOCIDO")
        despues_reemplazo = df[columna].str.contains("SIN DATO").sum()
        reemplazos_hechos = antes_reemplazo - despues_reemplazo
        print(f"Se han reemplazado {reemplazos_hechos} valores 'SIN DATO' por 'DESCONOCIDO' en la columna '{columna}'.")
        
        return df
    except Exception as e:
        print(f"Error al reemplazar los valores en la columna '{columna}': {e}")
        return df

    
def Estandarizacion_columnas(df, columnas):
   
    try:
        for columna in columnas:
            if columna in df.columns:
                # Aplicar la transformación
                df[columna] = df[columna].str.title()
                print(f"Columna '{columna}' transformada correctamente.")
            else:
                print(f"Columna '{columna}' no encontrada en el DataFrame.")
        return df
    except Exception as e:
        print(f"Error al transformar columnas: {e}")
        return df



'''
def agregar_region(df):
    """
    Agrega una columna de región basada en el departamento.
    """
    try:
        print("Agregando la columna REGIÓN...")

        # Diccionario de departamentos y regiones en Colombia
        departamento_a_region = {
            'Amazonas': 'Región Amazónica',
            'Antioquia': 'Región Andina',
            'Arauca': 'Región Orinoquía',
            'Atlántico': 'Región Caribe',
            'Bolívar': 'Región Caribe',
            'Boyacá': 'Región Andina',
            'Caldas': 'Región Andina',
            'Caquetá': 'Región Amazónica',
            'Casanare': 'Región Orinoquía',
            'Cauca': 'Región Pacífica',
            'Cesar': 'Región Caribe',
            'Chocó': 'Región Pacífica',
            'Córdoba': 'Región Caribe',
            'Cundinamarca': 'Región Andina',
            'Guainía': 'Región Amazónica',
            'Guaviare': 'Región Amazónica',
            'Huila': 'Región Andina',
            'La Guajira': 'Región Caribe',
            'Magdalena': 'Región Caribe',
            'Meta': 'Región Orinoquía',
            'Nariño': 'Región Pacífica',
            'Norte de Santander': 'Región Andina',
            'Putumayo': 'Región Amazónica',
            'Quindío': 'Región Andina',
            'Risaralda': 'Región Andina',
            'San Andrés, Providencia y Santa Catalina': 'Región Caribe',
            'Santander': 'Región Andina',
            'Sucre': 'Región Caribe',
            'Tolima': 'Región Andina',
            'Valle del Cauca': 'Región Pacífica',
            'Vaupés': 'Región Amazónica',
            'Vichada': 'Región Orinoquía',
            'BOGOTÁ, D. C.': 'Región Andina'
        }

        # Agregar la columna REGIÓN usando el mapeo
        df['REGIÓN'] = df['DEPARTAMENTO_HECHO'].map(departamento_a_region).fillna('Región Desconocida')

        print("Columna REGIÓN agregada con éxito.")
        print(df[['DEPARTAMENTO_HECHO', 'REGIÓN']].head())  # Mostrar ejemplos
        return df
    except Exception as e:
        print(f"Error al agregar la columna REGIÓN: {e}")
        return df

'''