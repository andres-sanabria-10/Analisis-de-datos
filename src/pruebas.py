import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from data_loader import load_data


# Ruta del CSV limpio
CLEAN_CSV_PATH = "data/victimas_sexuales_limpio.csv"


def prueba_z_proporciones(count1, nobs1, count2, nobs2):
    """
    Calcular prueba Z para comparación de proporciones
    
    Parámetros:
    count1, nobs1: número de éxitos y total de observaciones del primer grupo
    count2, nobs2: número de éxitos y total de observaciones del segundo grupo
    """
    # Calcular proporciones
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    
    # Proporción combinada
    p_pooled = (count1 + count2) / (nobs1 + nobs2)
    
    # Error estándar
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/nobs1 + 1/nobs2))
    
    # Estadístico Z
    z_stat = (p1 - p2) / se
    
    # Valor p (prueba bilateral)
    p_valor = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_valor

def comparar_proporciones(df):
    # Filtrar grupos etarios específicos
    grupo_ninos = df[df['GRUPO_ETARIO'] == 'Niño, Niña. Población de 0 a 13 años']
    grupo_adolescentes = df[df['GRUPO_ETARIO'] == 'Adolescente de 14 a 17 años']
    
    # Verificar que los grupos no estén vacíos
    if len(grupo_ninos) == 0 or len(grupo_adolescentes) == 0:
        print("Error: Uno o ambos grupos etarios están vacíos")
        return None
    
    # Contar víctimas por grupo etario
    victimas_ninos = grupo_ninos['TOTAL_VÍCTIMAS'].count()
    victimas_adolescentes = grupo_adolescentes['TOTAL_VÍCTIMAS'].count()
    
    # Total de registros por grupo etario
    total_ninos = len(grupo_ninos)
    total_adolescentes = len(grupo_adolescentes)
    
    # Calcular proporciones
    prop_ninos = victimas_ninos / total_ninos if total_ninos > 0 else 0
    prop_adolescentes = victimas_adolescentes / total_adolescentes if total_adolescentes > 0 else 0
    
    # Prueba Z para comparación de proporciones
    z_stat, p_valor = prueba_z_proporciones(
        victimas_ninos, total_ninos, 
        victimas_adolescentes, total_adolescentes
    )
    
    # Configuración de gráfico
    plt.figure(figsize=(10, 6), facecolor='white')
    
    # Gráfico de barras de proporciones
    grupos = ['Niños (0-13 años)', 'Adolescentes (14-17 años)']
    proporciones = [prop_ninos * 100, prop_adolescentes * 100]
    
    # Colores personalizados
    colors = ['#1f77b4', '#2ca02c']
    
    plt.bar(grupos, proporciones, color=colors, edgecolor='black', linewidth=1)
    plt.title('Proporción de Víctimas por Grupo Etario', fontsize=15, fontweight='bold')
    plt.ylabel('Porcentaje de Víctimas (%)', fontsize=12)
    plt.xlabel('Grupo Etario', fontsize=12)
    plt.ylim(0, max(proporciones) * 1.2)  # Ajustar límite superior
    
    # Añadir valores de porcentaje en las barras
    for i, v in enumerate(proporciones):
        plt.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Información estadística
    plt.text(0.5, -0.15, 
             f'Estadístico Z: {z_stat:.4f}\n'
             f'Valor p: {p_valor:.4f}', 
             horizontalalignment='center',
             verticalalignment='center', 
             transform=plt.gca().transAxes,
             fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resultados detallados
    print("\nResultados del Análisis:")
    print(f"Niños (0-13 años):")
    print(f"  Total registros: {total_ninos}")
    print(f"  Víctimas: {victimas_ninos}")
    print(f"  Proporción: {prop_ninos:.2%}")
    print(f"\nAdolescentes (14-17 años):")
    print(f"  Total registros: {total_adolescentes}")
    print(f"  Víctimas: {victimas_adolescentes}")
    print(f"  Proporción: {prop_adolescentes:.2%}")
    print(f"\nPrueba de Hipótesis:")
    print(f"  Estadístico Z: {z_stat:.4f}")
    print(f"  Valor p: {p_valor:.4f}")
    
    return {
        'prop_ninos': prop_ninos,
        'prop_adolescentes': prop_adolescentes,
        'z_stat': z_stat,
        'p_valor': p_valor
    }

# Ejemplo de uso (reemplazar con tu DataFrame real)
# Cargar los datos limpios
df = load_data(CLEAN_CSV_PATH)
resultados = comparar_proporciones(df)
print(resultados)