import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Iniciando el Análisis Exploratorio de Datos (EDA) para VITIS-IA...")

# 1. INGESTA Y UNIÓN DE LOS DATOS
try:
    df_xino = pd.read_excel('grape-maturity-dataset.xlsx', sheet_name='Xinomavro')
    df_syrah = pd.read_excel('grape-maturity-dataset.xlsx', sheet_name='Syrah')
    df_sauv = pd.read_excel('grape-maturity-dataset.xlsx', sheet_name='Sauvignon Blanc')
    print("Archivos CSV cargados correctamente.")
except FileNotFoundError as e:
    print(f"Error al cargar los archivos: {e}. Asegúrate de que están en la misma carpeta que el script.")
    exit()

# Añadir la columna de variedad
df_xino['Variedad'] = 'Xinomavro'
df_syrah['Variedad'] = 'Syrah'
df_sauv['Variedad'] = 'Sauvignon Blanc'

# Unir todo en un solo DataFrame
df = pd.concat([df_xino, df_syrah, df_sauv], ignore_index=True)

# (Nota: Si tu columna de azúcar se llama distinto a 'Brix', por ejemplo 'TSS', 
# cambia la palabra 'Brix' en las siguientes líneas por tu nombre real de columna)

# 2. GENERACIÓN DEL REPORTE EN TEXTO (.txt)
nombre_reporte = 'reporte_EDA_VITIS.txt'
with open(nombre_reporte, 'w', encoding='utf-8') as f:
    f.write("=========================================\n")
    f.write("REPORTE EDA - PROYECTO VITIS-IA\n")
    f.write("=========================================\n\n")
    
    f.write("1. DIMENSIONES DEL DATASET CONSOLIDADO\n")
    f.write(f"Total de filas: {df.shape[0]}\n")
    f.write(f"Total de columnas: {df.shape[1]}\n\n")
    
    f.write("2. VALORES NULOS POR COLUMNA\n")
    f.write(df.isnull().sum().to_string() + "\n\n")
    
    f.write("3. ESTADÍSTICAS DESCRIPTIVAS GLOBALES\n")
    f.write(df.describe().to_string() + "\n\n")
    
    # Comprobar si existe la columna Brix para dar estadísticas agrupadas
    col_azucar = 'Brix' # Cambia esto si en tu CSV se llama 'TSS' u otra cosa
    if col_azucar in df.columns:
        f.write("4. MEDIAS DE AZÚCAR POR VARIEDAD\n")
        f.write(df.groupby('Variedad')[col_azucar].mean().to_string() + "\n")

print(f"Reporte de texto generado y guardado como: {nombre_reporte}")

# 3. GENERACIÓN Y GUARDADO DE GRÁFICOS (.png)
# Configuramos Seaborn
sns.set_theme(style="whitegrid")

# Verificar que existen columnas numéricas para correlación
cols_numericas = df.select_dtypes(include=[np.number]).columns

if len(cols_numericas) > 1:
    # Gráfico 1: Matriz de Correlación
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[cols_numericas].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlación - Variables Bioquímicas')
    plt.tight_layout()
    plt.savefig('matriz_correlacion_vitis.png')
    plt.close()
    print("Gráfico guardado: matriz_correlacion_vitis.png")

if col_azucar in df.columns:
    # Gráfico 2: Distribución por Variedad (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Variedad', y=col_azucar, data=df, palette='Set2')
    plt.title('Distribución de Azúcar según Variedad')
    plt.ylabel('Grados Brix / TSS')
    plt.tight_layout()
    plt.savefig('distribucion_variedad_vitis.png')
    plt.close()
    print("Gráfico guardado: distribucion_variedad_vitis.png")

# Opcional: Gráfico temporal si tienes una columna de fecha (cambia 'Date' por tu nombre real)
col_fecha = 'Date' 
if col_fecha in df.columns and col_azucar in df.columns:
    # Convertimos a datetime para graficar mejor
    df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
    df = df.sort_values(by=col_fecha)
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=col_fecha, y=col_azucar, hue='Variedad', data=df, alpha=0.7)
    plt.title('Evolución Temporal de la Maduración')
    plt.xlabel('Fecha')
    plt.ylabel('Grados Brix / TSS')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('evolucion_temporal_vitis.png')
    plt.close()
    print("Gráfico guardado: evolucion_temporal_vitis.png")

print("\n¡Proceso finalizado con éxito! Revisa la carpeta para ver tu reporte y tus gráficos.")