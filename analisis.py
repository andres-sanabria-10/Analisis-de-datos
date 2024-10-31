import pandas as pd

# URL del archivo CSV en Dropbox (modificada para descarga)
url_procesos = 'https://www.dropbox.com/scl/fi/z3fojduofa0bztpcf9y77/Conteo_de_Procesos_V2_20241030.csv?rlkey=2plq8ayvm7ejnbz72n4qyppbf&dl=1'
url_indiciados = 'https://www.dropbox.com/scl/fi/wrpdzgn1ewdarnuy4xrxu/Conteo_de_Indiciados_V2_20241030.csv?rlkey=0m0z9udx0if0a9pr0ojrur5gf&dl=1'
#https://www.dropbox.com/scl/fi/z3fojduofa0bztpcf9y77/Conteo_de_Procesos_V2_20241030.csv?rlkey=2plq8ayvm7ejnbz72n4qyppbf&st=vxwge0ah&dl=0
#revisar de que forma cambian los enlaces
# Cargar los archivos CSV directamente desde la URL
datos5 = pd.read_csv(url_procesos, delimiter=',', on_bad_lines='skip', encoding='utf-8')
datos6 = pd.read_csv(url_indiciados, delimiter=',', on_bad_lines='skip', encoding='utf-8')

# Imprimir las columnas del DataFrame
print("Columnas de datos5:")
print(datos5.columns)

print("\nColumnas de datos6:")
print(datos6.columns)