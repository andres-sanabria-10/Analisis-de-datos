from dotenv import load_dotenv
from data_loader import load_data, upload_data_to_mongo
from data_preparation import clean_and_prepare_data, explore_data, agregar_region


from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Cargar variables de entorno
    load_dotenv()

    # Ruta del archivo CSV
    csv_path = "data/conteo_victimas_sexuales.csv"

    # Cargar datos
    df = load_data(csv_path)

    # Ver valores únicos de una columna específica
    #columna = 'AÑO_DENUNCIA'
    #valores_unicos = df[columna].unique()
    #print(f"Valores únicos en la columna '{columna}':\n{valores_unicos}")

    if df is not None:

        # Exploración inicial de los datos
        #explore_data(df)

        # Limpieza y transformación de los datos
        df = clean_and_prepare_data(df)

        # Agregar la columna REGIÓN
        #df = agregar_region(df)

        # Exploración despúes de la limpieza de los datos
        #explore_data(df)

        # Subir datos limpios a MongoDB
        #upload_data_to_mongo(df, "victimas")  # Nombre de la colección en MongoDB


if __name__ == "__main__":
    main()







    """
        # Ver las primeras filas
        #print(df.head())

        # Información general del dataset
        #print(df.info())



        # Ver valores únicos de una columna específica
        columna = 'ETAPA_CASO'
        valores_unicos = df[columna].unique()

        # Mostrar los valores únicos de forma enumerada
        print(f"\nValores únicos en la columna '{columna}':\n")
        for idx, valor in enumerate(valores_unicos, start=1):
            print(f"{idx}. {valor}")


        
        # Resumen estadístico (para columnas numéricas)
        print(df.describe())


        # Mostrar un resumen extendido
        columns_summary = {
            "Columnas": df.columns,
            "Tipo de dato": df.dtypes,
            "Valores nulos": df.isnull().sum(),
            "Valores únicos": df.nunique()
        }

        # Convertir a DataFrame para imprimir como tabla
        summary_df = pd.DataFrame(columns_summary)
        print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid'))


        print("\nValores faltantes por columna")
        # Valores faltantes por columna
        print(df.isnull().sum())

        print("\nIdentificar filas duplicadas")
        # Identificar filas duplicadas
        duplicated_rows = df[df.duplicated()]
        print(f"Filas duplicadas: {len(duplicated_rows)}")



       
        # Delitos más frecuentes por número total de víctimas
        delitos_frecuentes = df.groupby('DELITO')['TOTAL_VÍCTIMAS'].sum().sort_values(ascending=False).head(5)
        print(delitos_frecuentes)







        print("\nEstadísticas por año")
        # Estadísticas por año
        print(df.groupby('AÑO_DENUNCIA')['TOTAL_VÍCTIMAS'].sum().sort_values(ascending=False))

        print("\nEstadísticas por región (departamento)")
        # Estadísticas por región (departamento)
        print(df.groupby('DEPARTAMENTO_HECHO')['TOTAL_VÍCTIMAS'].sum().sort_values(ascending=False))


        # Top 10 delitos más comunes
        top_delitos = df['DELITO'].value_counts().head(10)
        top_delitos.plot(kind='bar', figsize=(10, 6))
        plt.title("Top 10 Delitos Más Comunes")
        plt.xlabel("Delitos")
        plt.ylabel("Frecuencia")
        plt.show()

        # Evolución de víctimas por año
        victimas_por_año = df.groupby('AÑO_HECHOS')['TOTAL_VÍCTIMAS'].sum()
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=victimas_por_año.index, y=victimas_por_año.values)
        plt.title("Evolución de Víctimas por Año")
        plt.xlabel("Año")
        plt.ylabel("Total de Víctimas")
        plt.show()

        """


        # Ver valores únicos de una columna específica
        #columna = 'SEXO'
        #valores_unicos = df[columna].unique()
        #print(f"Valores únicos en la columna '{columna}':\n{valores_unicos}")


