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
    total_victimas_por_sexo = total_victimas_por_sexo.sort_values(ascending=False)


    # Crear la gráfica
    plt.figure(figsize=(10, 8))
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
    
    if p_value < 0.05:
        print("Rechazamos la hipótesis nula: La proporción es significativamente diferente del 70%")
    else:
        print("No podemos rechazar la hipótesis nula: No hay evidencia suficiente para decir que la proporción es diferente del 70%")
    
    # Crear gráfico de torta
    proporciones = [p_observada * 100, (1 - p_observada) * 100]
    etiquetas = ["Comunidad LGTBI", "Otras comunidades"]
    
     # Mostrar resultados
    print("\nResultados de la Prueba Z para la Hipótesis 5:")
    print(f"Proporción observada LGTBI: {p_observada:.4f}")
    print(f"Proporción esperada: {p_esperada:.4f}")
    print(f"Estadístico Z: {z:.4f}")
    print(f"Valor p: {p_value:.4f}")
    
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
      
    if len(df_2024) == 0:
        print("No hay datos para el año 2024.")
        return
    
    # Filtrar víctimas de la comunidad LGTBI (valor "Si" en APLICA_LGBTI)
    victimas_lgtbi = df_2024[df_2024['APLICA_LGBTI'] == 'Si']
    total_lgtbi = victimas_lgtbi['TOTAL_VÍCTIMAS'].sum()
    
    
    if total_lgtbi == 0:
        print("No hay víctimas LGTBI en 2024.")
        return
    
    # Filtrar víctimas de las otras comunidades (filtrando por los valores específicos)
    victimas_indigena = df_2024[df_2024['INDÍGENA'] == 'Si']['TOTAL_VÍCTIMAS'].sum()
    victimas_afrodescendiente = df_2024[df_2024['AFRODESCENDIENTE'] == 'Si']['TOTAL_VÍCTIMAS'].sum()
    victimas_aplica_nna = df_2024[df_2024['APLICA_NNA'] == 'Si']['TOTAL_VÍCTIMAS'].sum()

# Sumar todas las víctimas de otras comunidades
    total_otras_comunidades = victimas_indigena + victimas_afrodescendiente + victimas_aplica_nna

    
    if total_otras_comunidades == 0:
        print("No hay víctimas en otras comunidades en 2024.")
        return
    
    # Calcular el total de víctimas
    total_victimas = total_lgtbi + total_otras_comunidades
    
    proporcion_otras_comunidades = total_otras_comunidades / total_victimas

    
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
    print(f"Proporción observada otras comunidades: {proporcion_otras_comunidades:.4f}")
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
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
        radius=1.1
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.title("Proporción de Víctimas por Comunidad (2024)", fontsize=16, fontweight='bold', pad=30)
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
    
    
    


def hipotesis_6():
    """
    Prueba Chi-cuadrado para validar la hipótesis:
    Existe una relación entre el año de denuncia (2022 y 2024) y la proporción de víctimas adultos mayores.
    """
    # Cargar los datos limpios
    df = load_data(CLEAN_CSV_PATH)
    
    # Verificar si las columnas necesarias existen
    if 'AÑO_DENUNCIA' not in df.columns or 'GRUPO_ETARIO' not in df.columns or 'TOTAL_VÍCTIMAS' not in df.columns:
        print("Las columnas necesarias ('AÑO_DENUNCIA', 'GRUPO_ETARIO', 'TOTAL_VÍCTIMAS') no están disponibles en los datos.")
        return
    
    # Asegurar que TOTAL_VÍCTIMAS sea numérica
    df['TOTAL_VÍCTIMAS'] = pd.to_numeric(df['TOTAL_VÍCTIMAS'], errors='coerce').fillna(0)
    
    # Filtrar datos para los años relevantes y asegurarse de crear una copia
    df_filtrado = df[df['AÑO_DENUNCIA'].isin([2022, 2024])].copy()
    
    if len(df_filtrado) == 0:
        print("No hay datos para los años 2022 y 2024.")
        return
    
    # Crear columna para identificar adultos mayores
    df_filtrado['ES_ADULTO_MAYOR'] = df_filtrado['GRUPO_ETARIO'].apply(
        lambda x: 'Sí' if 'Adulto Mayor. Personas Igual O Mayor A 60 Años.' in str(x) else 'No'
    )
    
    # Agrupar datos por año y categoría (adultos mayores o no)
    tabla_contingencia = df_filtrado.pivot_table(
        index='AÑO_DENUNCIA',
        columns='ES_ADULTO_MAYOR',
        values='TOTAL_VÍCTIMAS',
        aggfunc='sum',
        fill_value=0
    )

    # Realizar la prueba Chi-cuadrado
    chi2, p, dof, expected = chi2_contingency(tabla_contingencia)
    # Mostrar resultados
    print("\nResultados de la Prueba Chi-cuadrado:")
    print(f"Estadístico Chi-cuadrado: {chi2:.4f}")
    print(f"Grados de libertad: {dof}")
    print(f"Valor p: {p:.4f}")
    
    if p < 0.05:
        print("\nConclusión: Rechazamos la hipótesis nula. Existe una asociación significativa entre el año de denuncia y el grupo etario.")
    else:
        print("\nConclusión: No podemos rechazar la hipótesis nula. No hay evidencia suficiente para afirmar que existe una asociación significativa.")
    
    print("\nTabla de Contingencia:")
    print(tabla_contingencia)
    
    
  
    
    # Crear carpeta output si no existe
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Guardar gráfico de barras apiladas
    tabla_contingencia.plot(kind='bar', stacked=True, color=['#4A90E2', '#1F4788'], figsize=(10, 6))
    plt.title("Distribución de Víctimas Adultos Mayores y No Adultos Mayores por Año", fontsize=16, fontweight="bold")
    plt.xlabel("Año de Denuncia", fontsize=14)
    plt.ylabel("Total de Víctimas", fontsize=14)
    plt.legend(title="Es Adulto Mayor", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # Guardar gráfico de barras apiladas
    barras_path = os.path.join(output_dir, "hipotesis_adultos_mayores_barras.png")
    plt.savefig(barras_path, dpi=300)
    print(f"\nGráfico de barras guardado en: {barras_path}")
    
    plt.show()

    # Gráfico de líneas
    proporciones = tabla_contingencia.div(tabla_contingencia.sum(axis=1), axis=0)
    proporciones.plot(kind='line', marker='o', color=['#4A90E2', '#1F4788'], figsize=(10, 6))

    plt.title("Proporción de Víctimas Adultos Mayores y No Adultos Mayores por Año", fontsize=16, fontweight="bold")
    plt.xlabel("Año de Denuncia", fontsize=14)
    plt.ylabel("Proporción de Víctimas", fontsize=14)
    plt.legend(title="Es Adulto Mayor", fontsize=12)
    plt.grid(True, axis='y', linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Guardar gráfico de líneas
    lineas_path = os.path.join(output_dir, "hipotesis_adultos_mayores_lineas.png")
    plt.savefig(lineas_path, dpi=300)
    print(f"\nGráfico de líneas guardado en: {lineas_path}")
    
    plt.show()
    
    # Crear Heatmap (Mapa de Calor)
    plt.figure(figsize=(8, 6))
    sns.heatmap(tabla_contingencia, annot=True, cmap="Blues", fmt='g', cbar=True, annot_kws={"size": 16}, linewidths=0.5)
    plt.title("Mapa de Calor: Distribución de Víctimas por Año y Grupo Etario", fontsize=12, fontweight="bold")
    plt.xlabel("Grupo Etario", fontsize=14)
    plt.ylabel("Año de Denuncia", fontsize=14)
    plt.tight_layout()
    
    # Guardar el Heatmap
    heatmap_path = os.path.join(output_dir, "hipotesis_adultos_mayores_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    print(f"\nMapa de calor guardado en: {heatmap_path}")
    
    plt.show()
    
    # Visualización de resultados: Gráficos de torta para cada año
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for year in tabla_contingencia.index:
        # Extraer los valores para el gráfico de torta
        valores = tabla_contingencia.loc[year]
        etiquetas = ['No Adultos mayores de 60', 'Adultos Mayores de 60']
        colores = ['#4A90E2', '#1F4788']
        
        # Crear el gráfico de torta
        plt.figure(figsize=(6, 6))
        plt.pie(valores, labels=etiquetas, autopct='%1.1f%%', colors=colores, startangle=90, wedgeprops={'edgecolor': 'black'})
        plt.title(f"Distribución de Víctimas por Grupo Etario ({int(year)})", fontsize=12, fontweight="bold", pad=30)
        plt.axis('equal')  # Asegura que el gráfico sea circular
        
        # Guardar gráfico de torta
        torta_path = os.path.join(output_dir, f"hipotesis_adultos_mayores_torta_{int(year)}.png")
        plt.savefig(torta_path, dpi=300)
        print(f"\nGráfico de torta guardado en: {torta_path}")
        plt.show()





from scipy.stats import spearmanr, norm

def hipotesis_7():
    """
    Verificar si más del 50% de los casos denunciados en 2024 alcanzan la etapa de ejecución de penas.
    También calcula la correlación entre el año de denuncia y la etapa del caso (usando Spearman).
    """
    # Cargar los datos limpios
    df = load_data(CLEAN_CSV_PATH)
    
    # Eliminar espacios extra en los nombres de las columnas
    df.columns = df.columns.str.strip()
    
    # Verificar columnas necesarias
    required_columns = ['AÑO_DENUNCIA', 'ETAPA_CASO', 'TOTAL_VÍCTIMAS']
    if not all(col in df.columns for col in required_columns):
        print("Columnas necesarias no disponibles en los datos.")
        return
    
    # Asegurar que TOTAL_VICTIMAS sea numérica
    df['TOTAL_VÍCTIMAS'] = pd.to_numeric(df['TOTAL_VÍCTIMAS'], errors='coerce').fillna(0)
    
    # Crear una copia explícita para 2024
    df_2024 = df[df['AÑO_DENUNCIA'] == 2024].copy()
    
    if len(df_2024) == 0:
        print("No hay datos para el año 2024.")
        return
    
    # Calcular estadísticas para la prueba de hipótesis
    casos_ejecucion = df_2024[df_2024['ETAPA_CASO'].str.contains('Ejecución', na=False)]['TOTAL_VÍCTIMAS'].sum()
    total_casos = df_2024['TOTAL_VÍCTIMAS'].sum()
    
    # Cálculos estadísticos
    p_observada = casos_ejecucion / total_casos if total_casos > 0 else 0
    p_esperada = 0.5
    n = total_casos
    
    if n > 0:
        se = np.sqrt(p_esperada * (1 - p_esperada) / n)
        z = (p_observada - p_esperada) / se
        p_value = 1 - norm.cdf(z)
        
        # Imprimir resultados
        print("\nResultados de la Prueba Z para la hipótesis 7:")
        print(f"Número total de víctimas en 2024: {total_casos:,.0f}")
        print(f"Número de casos en ejecución de penas: {casos_ejecucion:,.0f}")
        print(f"Proporción observada: {p_observada:.4f}")
        print(f"Estadístico Z: {z:.4f}")
        print(f"Valor p: {p_value:.4f}")
        
        print("\nConclusión:", end=" ")
        if p_value < 0.05:
            print("Rechazamos la hipótesis nula.")
        else:
            print("No podemos rechazar la hipótesis nula.")
    
    # Codificación de etapas y correlación
    etapas_mapping = {
        'Indagación': 1,
        'Juicio': 2,
        'Ejecución De Penas': 3
    }
    
    # Crear columna ETAPA_CODIFICADA de manera segura
    df_2024['ETAPA_CODIFICADA'] = df_2024['ETAPA_CASO'].map(etapas_mapping)
    
    # Para la correlación, necesitamos más de un año
    df_todos = df.copy()
    df_todos['ETAPA_CODIFICADA'] = df_todos['ETAPA_CASO'].map(etapas_mapping)
    
    # Calcular correlación solo si hay suficiente variabilidad
    df_corr = df_todos.dropna(subset=['AÑO_DENUNCIA', 'ETAPA_CODIFICADA'])
    
    plt.figure(figsize=(15, 6))
    
    # Gráfico de Torta (izquierda)
    plt.subplot(1, 2, 1)
    etapas_counts = df_2024.groupby('ETAPA_CASO')['TOTAL_VÍCTIMAS'].sum()
    
    colores = ['#1F4788', '#4A90E2', '#2E6BAA']  # Azul oscuro, claro y medio
    
    plt.pie(etapas_counts, 
            labels=etapas_counts.index,
            colors=colores,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 7})
    
    plt.title('Distribución de Etapas de Casos (2024)', 
             fontsize=12, 
             fontweight='bold',
             pad=20)
    
    # Gráfico de Líneas (derecha)
    plt.subplot(1, 2, 2)
    tendencia_anual = df_todos.groupby(['AÑO_DENUNCIA', 'ETAPA_CASO'])['TOTAL_VÍCTIMAS'].sum().unstack()
    
    for columna, color in zip(tendencia_anual.columns, colores):
        plt.plot(tendencia_anual.index, 
                tendencia_anual[columna], 
                marker='o',
                color=color,
                label=columna,
                linewidth=2)
    
    plt.title('Evolución de Etapas por Año', 
             fontsize=12, 
             fontweight='bold',
             pad=20)
    
    plt.xlabel('Año de Denuncia', fontsize=10)
    plt.ylabel('Total de Víctimas', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar leyenda
    plt.legend(title='Etapa del Caso', 
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              fontsize=8,
              title_fontsize=9)
    
    # Rotar etiquetas del eje x
    plt.xticks(rotation=45)
    
    # Ajustar márgenes
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig('resultados_hipotesis_7_mejorado.png', 
                dpi=300, 
                bbox_inches='tight')
    
    plt.close()
    
    if len(df_corr['AÑO_DENUNCIA'].unique()) > 1:
        correlacion, p_spearman = spearmanr(df_corr['AÑO_DENUNCIA'], 
                                          df_corr['ETAPA_CODIFICADA'])
        
        print("\nCorrelación de Spearman (todos los años):")
        print(f"Coeficiente: {correlacion:.4f}")
        print(f"Valor p: {p_spearman:.4f}")
        
        print("\nConclusión correlación:", end=" ")
        if p_spearman < 0.05:
            print("Existe correlación significativa.")
        else:
            print("No hay evidencia de correlación significativa.")
    else:
        print("\nNo hay suficiente variabilidad en los años para calcular correlación.")


def main():
    """
    Función principal para ejecutar el análisis.
    """
   
    hipotesis_7()

if __name__ == "__main__":
    main()
