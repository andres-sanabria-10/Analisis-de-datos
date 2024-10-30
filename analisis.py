import pandas as pd
import gdown

# ID del archivo en Google Drive
file_id = '1erhDU8JPVFtFb7aFA2mA5oul8LM4Xov8'  # ID del archivo en tu URL

# Crear la URL de descarga
url = f"https://drive.google.com/uc?id={file_id}"

# Descargar el archivo
gdown.download(url, 'archivo.csv', quiet=False)

# Leer el archivo descargado en un DataFrame
datos5 = pd.read_csv('archivo.csv', encoding="utf-8")

# Filtrar los datos donde CRIMINALIDAD es "SI" y GRUPO_DELITO es "DELITOS SEXUALES"
datos_filtrados = datos5[(datos5['CRIMINALIDAD'] == "SI") & (datos5['GRUPO_DELITO'] == "DELITOS SEXUALES")]

# Mostrar los datos filtrados
print(datos_filtrados)

# Seleccionar solo la columna "DELITO"
filtrado_delitos = datos_filtrados[['DELITO']]
print(filtrado_delitos)
