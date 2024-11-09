import pandas as pd

def clean_and_transform_data(df):
    # Llenar valores nulos con 'NA'
    df.fillna("NA", inplace=True)
    
    # Convertir columnas de año a tipo numérico
    year_columns = ['AÑO_HECHOS', 'AÑO_ENTRADA', 'AÑO_DENUNCIA']
    for column in year_columns:
        # Intentar convertir a número; los errores se establecen como NaN para evitar errores en valores no convertibles
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    return df
