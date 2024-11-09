import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_data
from data_preparation import clean_and_transform_data 

def main():
    # Rutas de los archivos de datos
    victimas_path = 'data/conteo_victimas_sexuales.csv'
    victimas_cleaned_path = 'data/conteo_victimas_sexuales_cleaned.csv'  # Nuevo archivo para datos limpios

    # Cargar datos de víctimas
    df_victimas = load_data(victimas_path)

    # Limpiar y transformar los datos
    df_victimas = clean_and_transform_data(df_victimas)

    # Guardar los datos limpios en un nuevo archivo CSV
    df_victimas.to_csv(victimas_cleaned_path, index=False)  # `index=False` para no guardar el índice

    print(f"Archivo limpio guardado en: {victimas_cleaned_path}")

if __name__ == "__main__":
    main()



"""
    # Función para explorar cada columna de un dataframe
    def explore_dataframe(df, df_name):
        print(f"\nExploración del conjunto de datos: {df_name}")
        print("=" * 50)
        


        # Explorar los valores únicos de cada columna
        print("\nValores únicos y frecuencias de cada columna:")
        for column in df.columns:
            print(f"\nColumna: {column}")
            print(f"Valores únicos:\n{df[column].unique()}")

    # Ejecutar la exploración en ambos dataframes
    explore_dataframe(df_victimas, "Conteo de Víctimas")


    
    try:
        # Cargar datos de víctimas y procesos
        df_victimas, df_procesos = load_data(victimas_path, procesos_path)
        
        # Asegurarse de que 'AÑO_DENUNCIA' esté en formato numérico
        df_victimas['AÑO_DENUNCIA'] = pd.to_numeric(df_victimas['AÑO_DENUNCIA'], errors='coerce')
        df_procesos['AÑO_DENUNCIA'] = pd.to_numeric(df_procesos['AÑO_DENUNCIA'], errors='coerce')
        
        # Limpiar datos eliminando valores NaN en 'AÑO_DENUNCIA'
        df_victimas.dropna(subset=['AÑO_DENUNCIA'], inplace=True)
        df_procesos.dropna(subset=['AÑO_DENUNCIA'], inplace=True)
        
        # Convertir 'AÑO_DENUNCIA' a enteros
        df_victimas['AÑO_DENUNCIA'] = df_victimas['AÑO_DENUNCIA'].astype(int)
        df_procesos['AÑO_DENUNCIA'] = df_procesos['AÑO_DENUNCIA'].astype(int)

        # Calcular el total de víctimas y el total de procesos para 2024 y 2014
        for year in [2024, 2014]:
            victimas_year = df_victimas[df_victimas['AÑO_DENUNCIA'] == year]
            procesos_year = df_procesos[df_procesos['AÑO_DENUNCIA'] == year]
            
            total_victimas_year = victimas_year['TOTAL_VÍCTIMAS'].sum()
            total_procesos_year = procesos_year['TOTAL_PROCESOS'].sum()
            
            print(f"Total de víctimas en {year}: {total_victimas_year}")
            print(f"Total de procesos en {year}: {total_procesos_year}")
        
        # Filtrar los datos para obtener solo los registros del año 2024
        victimas_2024 = df_victimas[df_victimas['AÑO_DENUNCIA'] == 2024]
        procesos_2024 = df_procesos[df_procesos['AÑO_DENUNCIA'] == 2024]
        
        # Obtener los 3 departamentos con más víctimas y procesos en 2024
        top3_victimas_departamento = victimas_2024.groupby('DEPARTAMENTO_HECHO')['TOTAL_VÍCTIMAS'].sum().nlargest(3)
        top3_procesos_departamento = procesos_2024.groupby('DEPARTAMENTO_HECHO')['TOTAL_PROCESOS'].sum().nlargest(3)
        
        print("\nTop 3 departamentos con más víctimas en 2024:")
        print(top3_victimas_departamento)
        
        print("\nTop 3 departamentos con más procesos en 2024:")
        print(top3_procesos_departamento)
        
        # Cantidad de víctimas en 2024 por grupo etario
        victimas_grupo_etario_2024 = victimas_2024.groupby('GRUPO_ETARIO')['TOTAL_VÍCTIMAS'].sum()
        menores_18 = victimas_grupo_etario_2024.get('Niño, Niña. Población de 0 a 13 años.', 0) + victimas_grupo_etario_2024.get('Adolescente de 14 a 17 años.', 0)
        mayores_18 = victimas_grupo_etario_2024.sum() - menores_18
        
        print("\nMenores de 18 en 2024:")
        print(menores_18)
        print(mayores_18)
        print("\nCantidad de víctimas en 2024 por grupo etario:")
        print(victimas_grupo_etario_2024)
        
        # Cantidad de víctimas en 2024 por sexo
        victimas_sexo_2024 = victimas_2024.groupby('SEXO')['TOTAL_VÍCTIMAS'].sum()
        
        print("\nCantidad de víctimas en 2024 por sexo:")
        print(victimas_sexo_2024)
        
        # Colores personalizados para los gráficos de torta
        colors_age = ['#FF6F61', '#6B5B95']
        colors_gender = ['#FF6F61', '#6B5B95', '#88B04B']

        # Gráfico de torta para grupo etario (menores vs. mayores de 18)
        plt.figure(figsize=(8, 6))
        plt.pie([menores_18, mayores_18], labels=['Menores de 18 años', '18 años o más'], 
                autopct='%1.1f%%', startangle=90, colors=colors_age, explode=[0.05, 0], 
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        plt.title('Distribución de Víctimas en 2024 por Edad', fontsize=16, fontweight='bold')
        plt.show()
        
        # Gráfico de torta para sexo
        plt.figure(figsize=(8, 6))
        plt.pie([victimas_sexo_2024.get('FEMENINO', 0), victimas_sexo_2024.get('MASCULINO', 0), victimas_sexo_2024.get('SIN DATO', 0)], 
                labels=['Femenino', 'Masculino', 'Sin Dato'], autopct='%1.1f%%', startangle=90, 
                colors=colors_gender, explode=[0.05, 0, 0], 
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        plt.title('Distribución de Víctimas en 2024 por Sexo', fontsize=16, fontweight='bold')
        plt.show()

    except Exception as e:
        print(f"Error en la carga de datos: {str(e)}") 
        """