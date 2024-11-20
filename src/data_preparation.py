import pandas as pd

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
