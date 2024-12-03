import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_data
from scipy.stats import chi2_contingency
import numpy as np
from scipy.stats import norm
from scipy import stats


# Ruta del CSV limpio
CLEAN_CSV_PATH = "data/victimas_sexuales_limpio.csv"



def hipotesis_1():
    """
    Análisis para determinar si el número total de víctimas de sexo femenino 
    es mayor que el de sexo masculino en delitos sexuales.
    """
    # Cargar los datos limpios
    df = load_data(CLEAN_CSV_PATH)

    # Verificar si las columnas necesarias existen
    if 'SEXO' not in df.columns or 'TOTAL_VÍCTIMAS' not in df.columns:
        print("Las columnas 'SEXO' y/o 'TOTAL_VÍCTIMAS' no están disponibles en los datos.")
        return

    # Asegurar que TOTAL_VÍCTIMAS sea numérica
    df['TOTAL_VÍCTIMAS'] = pd.to_numeric(df['TOTAL_VÍCTIMAS'], errors='coerce').fillna(0)

    # Sumar el total de víctimas por sexo
    total_victimas_por_sexo = df.groupby('SEXO')['TOTAL_VÍCTIMAS'].sum()

    # Mostrar resultados en consola
    print("\nTotal de víctimas por sexo:")
    print(total_victimas_por_sexo)

    # Crear la gráfica
    plt.figure(figsize=(10, 7))
    bars = sns.barplot(
        x=total_victimas_por_sexo.index,
        y=total_victimas_por_sexo.values,
        palette=["#1F4788", "#4A90E2", "#87CEEB"]  # Colores personalizados
    )
    bars.set_title("Número Total de Víctimas por Sexo en Delitos Sexuales", fontsize=18, fontweight="bold")
    bars.set_xlabel("Sexo", fontsize=14, labelpad=15)
    bars.set_ylabel("Total de Víctimas", fontsize=14, labelpad=15)
    bars.set_xticklabels(total_victimas_por_sexo.index, fontsize=12)

    # Ajustar el rango del eje Y para que no se vea tan acumulado
    max_value = max(total_victimas_por_sexo.values)
    step = 50000  # Intervalos de las marcas del eje y
    bars.set_yticks(range(0, int(max_value) + step, step))

    # Agregar etiquetas a las barras
    for i, value in enumerate(total_victimas_por_sexo.values):
        plt.text(
            i, value + 5000, f"{value:,.0f}",
            ha='center', va='bottom', fontsize=12, color='black', fontweight='bold'
        )

    # Crear carpeta output si no existe
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Guardar la gráfica
    output_image_path = os.path.join(output_dir, "analisis_total_victimas_sexo_mejorado.png")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"Gráfica guardada en: {output_image_path}")

    # Mostrar la gráfica
    plt.show()


def hipotesis_2():
    """
    Análisis para determinar si más del 30% de las denuncias realizadas en el año 2024 
    corresponden a hechos ocurridos hace más de un año, basado en la suma del total de víctimas.
    """
    # Cargar los datos limpios
    df = load_data(CLEAN_CSV_PATH)

    # Verificar si las columnas necesarias existen
    if 'AÑO_HECHOS' not in df.columns or 'AÑO_DENUNCIA' not in df.columns or 'TOTAL_VÍCTIMAS' not in df.columns:
        print("Las columnas necesarias ('AÑO_HECHOS', 'AÑO_DENUNCIA', 'TOTAL_VÍCTIMAS') no están disponibles en los datos.")
        return

    # Asegurar que TOTAL_VÍCTIMAS sea numérica
    df['TOTAL_VÍCTIMAS'] = pd.to_numeric(df['TOTAL_VÍCTIMAS'], errors='coerce').fillna(0)

    # Filtrar las denuncias realizadas en el año 2024
    df_2024 = df[df['AÑO_DENUNCIA'] == 2024]

    # Categorizar las denuncias según el año de los hechos
    df_2024['Hechos_2024'] = df_2024['AÑO_HECHOS'].apply(lambda x: 'Hechos 2024' if x == 2024 else 'Hechos antes de 2024')

    # Calcular la suma del total de víctimas en cada categoría
    victimas_por_categoria = df_2024.groupby('Hechos_2024')['TOTAL_VÍCTIMAS'].sum()

    # Calcular la proporción de denuncias de hechos ocurridos antes de 2024
    proporcion_hechos_pasados = victimas_por_categoria['Hechos antes de 2024'] / victimas_por_categoria.sum() * 100

    # Mostrar resultados
    print(f"\nProporción de víctimas de hechos ocurridos antes de 2024: {proporcion_hechos_pasados:.2f}%")
    print(f"Total víctimas en denuncias 2024: {victimas_por_categoria.sum()}")
    print(f"Total víctimas en hechos 2024: {victimas_por_categoria['Hechos 2024']}")
    print(f"Total víctimas en hechos antes de 2024: {victimas_por_categoria['Hechos antes de 2024']}")

    # Crear gráfico de torta
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        victimas_por_categoria, 
        labels=victimas_por_categoria.index, 
        autopct=lambda pct: f'{pct:.1f}%',  # Formato del porcentaje
        startangle=90, 
        colors=["#1F4788", "#4A90E2"],  # Colores personalizados
        explode=(0.1, 0),  # Explode la primera porción
        wedgeprops={'edgecolor': 'black'}
    )

    # Aumentar el tamaño y poner en negrilla el texto del porcentaje
    for autotext in autotexts:
        autotext.set_fontsize(12)  # Aumentar el tamaño
        autotext.set_fontweight('bold')  # Poner en negrilla

    # Título y ajustes visuales
    plt.title('Proporción de Víctimas: Hechos 2024 vs Hechos Anteriores', fontsize=16, fontweight='bold')
    plt.axis('equal')  # Para que el gráfico sea circular
    plt.tight_layout()

    # Crear carpeta output si no existe
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Guardar la gráfica
    output_image_path = os.path.join(output_dir, "analisis_denuncias_hechos_pasados_2024.png")
    plt.savefig(output_image_path, dpi=300)
    print(f"Gráfica de análisis guardada en: {output_image_path}")

    # Mostrar la gráfica
    plt.show()



def hipotesis_3():
    """
    Hipótesis 3: Las víctimas que no tienen como país de nacimiento 'Colombia' tienen un menor porcentaje de casos resueltos.
    
    Retorna:
    dict: Resultados del análisis
    """
    # Cargar los datos limpios
    df = load_data(CLEAN_CSV_PATH)
    
    # Crear una categoría binaria de resolución basada en ETAPA_CASO
    df['caso_resuelto'] = df['ETAPA_CASO'].apply(
        lambda x: 'Resuelto' if x in ['Ejecución De Penas', 'Juicio'] else 'No Resuelto'
    )
    
    # Crear una columna binaria para país de nacimiento (Colombia vs. Otros)
    df['pais_nacimiento_colombiano'] = (df['PAÍS_NACIMIENTO_VICTIMA'] == 'Colombia').astype(int)
    
    # Filtrar datos para Colombia y otros países
    colombia_df = df[df['pais_nacimiento_colombiano'] == 1]
    otros_paises_df = df[df['pais_nacimiento_colombiano'] == 0]
    
    # Calcular el total de víctimas por país de nacimiento
    total_colombia = colombia_df['TOTAL_VÍCTIMAS'].sum()
    total_otros_paises = otros_paises_df['TOTAL_VÍCTIMAS'].sum()

    # Calcular la proporción de "Resuelto" ponderada por el total de víctimas
    colombia_resuelto = colombia_df.groupby('caso_resuelto')['TOTAL_VÍCTIMAS'].sum()
    otros_paises_resuelto = otros_paises_df.groupby('caso_resuelto')['TOTAL_VÍCTIMAS'].sum()

    # Proporción de casos resueltos
    colombia_proporcion = colombia_resuelto['Resuelto'] / total_colombia * 100
    otros_paises_proporcion = otros_paises_resuelto['Resuelto'] / total_otros_paises * 100
    
    # Mostrar los resultados
    print(f"\nProporción de casos resueltos en Colombia: {colombia_proporcion:.2f}%")
    print(f"Proporción de casos resueltos en Otros Países: {otros_paises_proporcion:.2f}%")
    print(f"Total de víctimas en Colombia: {total_colombia}")
    print(f"Total de víctimas en Otros Países: {total_otros_paises}")
    
    # Crear los gráficos de torta
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Gráfico de torta para Colombia
    wedges, texts, autotexts = axes[0].pie(
        colombia_resuelto, 
        labels=colombia_resuelto.index, 
        autopct='%1.1f%%', 
        colors=['#1F4788', '#4A90E2'], 
        startangle=90, 
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )
    
    # Personalizar los porcentajes en negrita
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    axes[0].set_title(f'Distribución de Resolución de Casos para victimas\nNacidas en Colombia', fontsize=14, fontweight='bold')
    axes[0].axis('equal')  # Asegurar que el gráfico de torta sea circular
    
    # Gráfico de torta para Otros Países
    wedges, texts, autotexts = axes[1].pie(
        otros_paises_resuelto, 
        labels=otros_paises_resuelto.index, 
        autopct='%1.1f%%', 
        colors=['#1F4788', '#4A90E2'], 
        startangle=90, 
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )
    
    # Personalizar los porcentajes en negrita
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    axes[1].set_title(f'Distribución de Resolución de Casos para victimas\n Nacidas en Otros Países', fontsize=14, fontweight='bold')
    axes[1].axis('equal')  # Asegurar que el gráfico de torta sea circular
    
    # Mostrar los gráficos de torta
    plt.tight_layout()
    plt.show()
    
    # Mapa de calor para correlación de resolución de casos por país de nacimiento
    crosstab = pd.crosstab(df['pais_nacimiento_colombiano'], df['caso_resuelto'], values=df['TOTAL_VÍCTIMAS'], aggfunc='sum', normalize='index') * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        crosstab, 
        annot=True, 
        cmap='coolwarm', 
        fmt='.1f', 
        linewidths=0.5, 
        cbar_kws={'label': 'Porcentaje'}
    )
    plt.title('Resolución de Casos por País de Nacimiento', fontsize=14, fontweight='bold')
    plt.xlabel('Estado de Resolución')
    plt.ylabel('País de Nacimiento (0 = Otros, 1 = Colombia)')
    plt.tight_layout()
    plt.show()

    # Prueba estadística (Chi-cuadrado)
    contingency_table = pd.crosstab(df['pais_nacimiento_colombiano'], df['caso_resuelto'], values=df['TOTAL_VÍCTIMAS'], aggfunc='sum')
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Mostrar el valor de chi-cuadrado y el valor p
    print("\nPrueba estadística (Chi-cuadrado):")
    print(f"Chi-cuadrado: {chi2}")
    print(f"Valor p: {p_value}")
    print(f"Grados de libertad: {dof}")

    # Hipótesis
    if p_value < 0.05:
        print("Rechazamos la hipótesis nula: Existe una diferencia significativa entre los dos grupos.")
    else:
        print("No rechazamos la hipótesis nula: No hay una diferencia significativa entre los dos grupos.")
    
    # Imprimir resultados
    print("\nProporción de Casos Resueltos en Colombia:")
    print(colombia_proporcion)
    print("\nProporción de Casos Resueltos en Otros Países:")
    print(otros_paises_proporcion)



def hipotesis_4():
    """
    Prueba Z para validar la hipótesis 5:
    Las víctimas del grupo etario “Niño, Niña. Población de 0 a 13 años” son mayores
    en comparación con el grupo etario “Adolescente de 14 a 17 años”.
    """
    # Cargar los datos limpios
    df = load_data(CLEAN_CSV_PATH)

    # Verificar si las columnas necesarias existen
    if 'GRUPO_ETARIO' not in df.columns or 'TOTAL_VÍCTIMAS' not in df.columns:
        print("Las columnas necesarias ('GRUPO_ETARIO', 'TOTAL_VÍCTIMAS') no están disponibles en los datos.")
        return

    # Asegurar que TOTAL_VÍCTIMAS sea numérica
    df['TOTAL_VÍCTIMAS'] = pd.to_numeric(df['TOTAL_VÍCTIMAS'], errors='coerce').fillna(0)

    # Filtrar los grupos etarios de interés
    grupos_interes = [
        "Niño, Niña. Población De 0 A 13 Años.",
        "Adolescente De 14 A 17 Años."
    ]
    df_grupos = df[df['GRUPO_ETARIO'].isin(grupos_interes)]

    # Sumar el total de víctimas por grupo etario
    victimas_por_grupo = df_grupos.groupby('GRUPO_ETARIO')['TOTAL_VÍCTIMAS'].sum()
    total_0_13 = victimas_por_grupo["Niño, Niña. Población De 0 A 13 Años."]
    total_14_17 = victimas_por_grupo["Adolescente De 14 A 17 Años."]
    total_victimas = victimas_por_grupo.sum()

    # Calcular proporciones
    p1 = total_0_13 / total_victimas
    p2 = total_14_17 / total_victimas

    # Calcular error estándar
    se = np.sqrt((p1 * (1 - p1)) / total_0_13 + (p2 * (1 - p2)) / total_14_17)

    # Calcular estadístico Z
    z = (p1 - p2) / se

    # Calcular valor p (prueba de una cola: p1 > p2)
    p_value = 1 - norm.cdf(z)

    # Mostrar resultados
    print("\nResultados de la Prueba Z para la Hipótesis 4:")
    print(f"Proporción 0-13 años: {p1:.4f}")
    print(f"Proporción 14-17 años: {p2:.4f}")
    print(f"Estadístico Z: {z:.4f}")
    print(f"Valor p: {p_value:.4f}")

    if p_value < 0.05:
        print("Rechazamos la hipótesis nula: Existe una diferencia significativa entre los grupos.")
    else:
        print("No podemos rechazar la hipótesis nula: No hay evidencia suficiente de una diferencia significativa.")

    # Crear gráfico de torta
    proporciones = [p1 * 100, p2 * 100]
    etiquetas = ["Niño, Niña (0-13 años)", "Adolescente (14-17 años)"]

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        proporciones,
        labels=etiquetas,
        autopct='%1.1f%%',
        colors=["#1F4788", "#4A90E2"],
        startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    plt.title("Proporción de Víctimas por Grupo Etario", fontsize=16, fontweight='bold')
    plt.axis('equal')  # Asegurar que el gráfico sea circular
    plt.tight_layout()

    # Crear carpeta output si no existe
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Guardar gráfico de torta
    torta_path = os.path.join(output_dir, "hipotesis_5_torta.png")
    plt.savefig(torta_path, dpi=300)
    print(f"Gráfico de torta guardado en: {torta_path}")

    plt.show()

    # Crear gráfico de barras con etiquetas de conteo en la parte superior
    plt.figure(figsize=(10, 6))
    totales = [total_14_17, total_0_13]  # Usamos los totales de cada grupo
    etiquetas_barras = ["Adolescente De 14 A 17 Años.", "Niño, Niña. Población De 0 A 13 Años."]

    # Colores personalizados
    colores = ["#4A90E2", "#1F4788"]

    # Crear gráfico
    barras = plt.bar(etiquetas_barras, totales, color=colores, alpha=0.9)

    # Agregar etiquetas en la parte superior de cada barra
    for i, barra in enumerate(barras):
        plt.text(
            barra.get_x() + barra.get_width() / 2,  # Posición horizontal
            barra.get_height() + 5000,  # Posición vertical ligeramente encima de la barra
            f"{int(totales[i]):,}".replace(",", "."),  # Formato de número con puntos
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    # Personalizar el gráfico
    plt.title("Total de Víctimas por Grupo Etario", fontsize=16, fontweight="bold")
    plt.xlabel("Grupo Etario", fontsize=14)
    plt.ylabel("Total de Víctimas", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, max(totales) + 30000)  # Ajustar el límite superior del eje Y

    plt.tight_layout()

    # Guardar gráfico de barras
    barras_path = os.path.join(output_dir, "hipotesis_5_barras_actualizado.png")
    plt.savefig(barras_path, dpi=300)
    print(f"Gráfico de barras actualizado guardado en: {barras_path}")

    plt.show()
    
    #hipotesis 5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

def hipotesis_5():
    """
    Prueba Z para validar la hipótesis 5:
    El 70% de las víctimas hacen parte de la comunidad LGTBI en comparación
    a las otras comunidades ("INDÍGENAS", "AFRODESCENDIENTE", "APLICA_NNA") para el año de denuncia 2024.
    """
    # Cargar los datos limpios
    df = load_data(CLEAN_CSV_PATH)
    
    # Verificar si las columnas necesarias existen
    if 'APLICA_LGBTI' not in df.columns or 'TOTAL_VÍCTIMAS' not in df.columns or 'AÑO_DENUNCIA' not in df.columns:
        print("Las columnas necesarias ('APLICA_LGBTI', 'TOTAL_VÍCTIMAS', 'AÑO_DENUNCIA') no están disponibles en los datos.")
        return
    
    # Asegurar que TOTAL_VÍCTIMAS sea numérica
    df['TOTAL_VÍCTIMAS'] = pd.to_numeric(df['TOTAL_VÍCTIMAS'], errors='coerce').fillna(0)
    
    # Filtrar datos para 2024
    df_2024 = df[df['AÑO_DENUNCIA'] == 2024]
    
    # Filtrar víctimas de la comunidad LGTBI
    total_lgtbi = df_2024[df_2024['APLICA_LGBTI'] == 'LGTBI']['TOTAL_VÍCTIMAS'].sum()
    
    # Filtrar víctimas de las otras comunidades: "INDÍGENAS", "AFRODESCENDIENTE", "APLICA_NNA"
    otras_comunidades = df_2024[df_2024['APLICA_LGBTI'].isin(['INDÍGENAS', 'AFRODESCENDIENTE', 'APLICA_NNA'])]
    total_otras_comunidades = otras_comunidades['TOTAL_VÍCTIMAS'].sum()
    
    # Calcular el total de víctimas
    total_victimas = total_lgtbi + total_otras_comunidades
    
    # Calcular proporción observada de víctimas LGTBI
    p_observada = total_lgtbi / total_victimas
    p_esperada = 0.70  # Proporción esperada según la hipótesis
    
    # Calcular error estándar
    se = np.sqrt((p_esperada * (1 - p_esperada)) / total_victimas)
    
    # Calcular estadístico Z
    z = (p_observada - p_esperada) / se
    
    # Calcular valor p (prueba de dos colas)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    # Mostrar resultados
    print("\nResultados de la Prueba Z para la Hipótesis 5:")
    print(f"Proporción observada LGTBI: {p_observada:.4f}")
    print(f"Proporción esperada: {p_esperada:.4f}")
    print(f"Estadístico Z: {z:.4f}")
    print(f"Valor p: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Rechazamos la hipótesis nula: La proporción es significativamente diferente del 70%")
    else:
        print("No podemos rechazar la hipótesis nula: No hay evidencia suficiente para decir que la proporción es diferente del 70%")
    
    # Crear gráfico de torta
    proporciones = [p_observada * 100, (1 - p_observada) * 100]
    etiquetas = ["Comunidad LGTBI", "Otras comunidades"]
    
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        proporciones,
        labels=etiquetas,
        autopct='%1.1f%%',
        colors=["#1F4788", "#4A90E2"],
        startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.title("Proporción de Víctimas por Comunidad (2024)", fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    
    # Crear carpeta output si no existe
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Guardar gráfico de torta
    torta_path = os.path.join(output_dir, "hipotesis_5_torta_lgtbi.png")
    plt.savefig(torta_path, dpi=300)
    print(f"Gráfico de torta guardado en: {torta_path}")
    
    plt.show()
    
    # Crear gráfico de barras
    plt.figure(figsize=(10, 6))
    totales = [total_lgtbi, total_otras_comunidades]
    etiquetas_barras = ["Comunidad LGTBI", "Otras comunidades"]
    
    # Colores personalizados
    colores = ["#4A90E2", "#1F4788"]
    
    # Crear gráfico
    barras = plt.bar(etiquetas_barras, totales, color=colores, alpha=0.9)
    
    # Agregar etiquetas en la parte superior de cada barra
    for i, barra in enumerate(barras):
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + (max(totales) * 0.02),
            f"{int(totales[i]):,}".replace(",", "."), ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
    
    # Personalizar el gráfico
    plt.title("Total de Víctimas por Comunidad (2024)", fontsize=16, fontweight="bold")
    plt.xlabel("Comunidad", fontsize=14)
    plt.ylabel("Total de Víctimas", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, max(totales) * 1.1)
    
    plt.tight_layout()
    
    # Guardar gráfico de barras
    barras_path = os.path.join(output_dir, "hipotesis_5_barras_lgtbi.png")
    plt.savefig(barras_path, dpi=300)
    print(f"Gráfico de barras guardado en: {barras_path}")
    
    plt.show()


def hipotesis_5():
    """
    Prueba Z para validar la hipótesis 5:
    El 70% de las víctimas hacen parte de la comunidad LGTBI en comparación
    a las otras comunidades ("INDÍGENAS", "AFRODESCENDIENTE", "APLICA_NNA") para el año de denuncia 2024.
    """
    # Cargar los datos limpios
    df = load_data(CLEAN_CSV_PATH)
    
    # Verificar si las columnas necesarias existen
    if 'APLICA_LGBTI' not in df.columns or 'TOTAL_VÍCTIMAS' not in df.columns or 'AÑO_DENUNCIA' not in df.columns:
        print("Las columnas necesarias ('APLICA_LGBTI', 'TOTAL_VÍCTIMAS', 'AÑO_DENUNCIA') no están disponibles en los datos.")
        return
    
    # Asegurar que TOTAL_VÍCTIMAS sea numérica
    df['TOTAL_VÍCTIMAS'] = pd.to_numeric(df['TOTAL_VÍCTIMAS'], errors='coerce').fillna(0)
    
    # Filtrar datos para 2024
    df_2024 = df[df['AÑO_DENUNCIA'] == 2024]
    
    # Depuración: Verificar cuántos registros hay en 2024
    print(f"Cantidad de registros para 2024: {len(df_2024)}")
    
    if len(df_2024) == 0:
        print("No hay datos para el año 2024.")
        return
    
    # Filtrar víctimas de la comunidad LGTBI (valor "Si" en APLICA_LGBTI)
    victimas_lgtbi = df_2024[df_2024['APLICA_LGBTI'] == 'Si']
    total_lgtbi = victimas_lgtbi['TOTAL_VÍCTIMAS'].sum()
    
    # Depuración: Verificar las víctimas de LGTBI
    print(f"Total de víctimas LGTBI: {total_lgtbi}")
    
    if total_lgtbi == 0:
        print("No hay víctimas LGTBI en 2024.")
        return
    
    # Filtrar víctimas de las otras comunidades (filtrando por los valores específicos)
    otras_comunidades = df_2024[df_2024['APLICA_LGBTI'].isin(['INDÍGENA', 'AFRODESCENDIENTE', 'APLICA_NNA'])]
    total_otras_comunidades = otras_comunidades['TOTAL_VÍCTIMAS'].sum()
    
    # Depuración: Verificar las víctimas de otras comunidades
    print(f"Total de víctimas en otras comunidades: {total_otras_comunidades}")
    
    if total_otras_comunidades == 0:
        print("No hay víctimas en otras comunidades en 2024.")
        return
    
    # Calcular el total de víctimas
    total_victimas = total_lgtbi + total_otras_comunidades
    
    # Calcular proporción observada de víctimas LGTBI
    p_observada = total_lgtbi / total_victimas
    p_esperada = 0.70  # Proporción esperada según la hipótesis
    
    # Calcular error estándar
    se = np.sqrt((p_esperada * (1 - p_esperada)) / total_victimas)
    
    # Calcular estadístico Z
    z = (p_observada - p_esperada) / se
    
    # Calcular valor p (prueba de dos colas)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    # Mostrar resultados
    print("\nResultados de la Prueba Z para la Hipótesis 5:")
    print(f"Proporción observada LGTBI: {p_observada:.4f}")
    print(f"Proporción esperada: {p_esperada:.4f}")
    print(f"Estadístico Z: {z:.4f}")
    print(f"Valor p: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Rechazamos la hipótesis nula: La proporción es significativamente diferente del 70%")
    else:
        print("No podemos rechazar la hipótesis nula: No hay evidencia suficiente para decir que la proporción es diferente del 70%")
    
    # Crear gráfico de torta
    proporciones = [p_observada * 100, (1 - p_observada) * 100]
    etiquetas = ["Comunidad LGTBI", "Otras comunidades"]
    
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        proporciones,
        labels=etiquetas,
        autopct='%1.1f%%',
        colors=["#1F4788", "#4A90E2"],
        startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.title("Proporción de Víctimas por Comunidad (2024)", fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    
    # Crear carpeta output si no existe
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Guardar gráfico de torta
    torta_path = os.path.join(output_dir, "hipotesis_5_torta_lgtbi.png")
    plt.savefig(torta_path, dpi=300)
    print(f"Gráfico de torta guardado en: {torta_path}")
    
    plt.show()
    
    # Crear gráfico de barras
    plt.figure(figsize=(10, 6))
    totales = [total_lgtbi, total_otras_comunidades]
    etiquetas_barras = ["Comunidad LGTBI", "Otras comunidades"]
    
    # Colores personalizados
    colores = ["#4A90E2", "#1F4788"]
    
    # Crear gráfico
    barras = plt.bar(etiquetas_barras, totales, color=colores, alpha=0.9)
    
    # Agregar etiquetas en la parte superior de cada barra
    for i, barra in enumerate(barras):
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + (max(totales) * 0.02),
            f"{int(totales[i]):,}".replace(",", "."), ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
    
    # Personalizar el gráfico
    plt.title("Total de Víctimas por Comunidad (2024)", fontsize=16, fontweight="bold")
    plt.xlabel("Comunidad", fontsize=14)
    plt.ylabel("Total de Víctimas", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, max(totales) * 1.1)
    
    plt.tight_layout()
    
    # Guardar gráfico de barras
    barras_path = os.path.join(output_dir, "hipotesis_5_barras_lgtbi.png")
    plt.savefig(barras_path, dpi=300)
    print(f"Gráfico de barras guardado en: {barras_path}")
    
    plt.show()


def main():
    """
    Función principal para ejecutar el análisis.
    """
    print("Iniciando análisis para la hipótesis 1...")
    hipotesis_5()

if __name__ == "__main__":
    main()
