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
    print("Archivos Excel de uvas cargados correctamente.")
except FileNotFoundError as e:
    print(f"Error al cargar excel de uvas: {e}. Asegúrate de que están en la misma carpeta.")
    exit()

# Añadir la columna de variedad
df_xino['Variedad'] = 'Xinomavro'
df_syrah['Variedad'] = 'Syrah'
df_sauv['Variedad'] = 'Sauvignon Blanc'

# Unir todo en un solo DataFrame de uvas
df = pd.concat([df_xino, df_syrah, df_sauv], ignore_index=True)

try:
    df_rosales = pd.read_csv('plant_health_data.csv')
    print("Archivo CSV de rosales cargado correctamente.")
except FileNotFoundError:
    print("Aviso: No se encontró 'plant_health_data.csv'. Se omitirá esta parte.")
    df_rosales = None


# 2. GENERACIÓN DEL REPORTE EN TEXTO (.txt)
nombre_reporte = 'reporte_EDA_VITIS.txt'
with open(nombre_reporte, 'w', encoding='utf-8') as f:
    f.write("=========================================\n")
    f.write("REPORTE EDA - PROYECTO VITIS-IA\n")
    f.write("=========================================\n\n")
    
    f.write("--- DATOS DE MADURACIÓN (UVAS) ---\n")
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
        f.write(df.groupby('Variedad')[col_azucar].mean().to_string() + "\n\n")

    if df_rosales is not None and not df_rosales.empty:
        f.write("=========================================\n")
        f.write("--- DATOS DE FITOSANIDAD (ROSALES) ---\n")
        f.write(f"Total de filas: {df_rosales.shape[0]} | Columnas: {df_rosales.shape[1]}\n\n")
        
        f.write("VALORES NULOS POR COLUMNA\n")
        f.write(df_rosales.isnull().sum().to_string() + "\n\n")
        
        # Análisis de sesgo/desbalanceo de la clase
        col_plaga = 'Plant_Health_Status' 
        if col_plaga in df_rosales.columns:
            f.write("DESBALANCEO DE CLASES (VARIABLE OBJETIVO)\n")
            f.write("Proporción de días sanos vs con plaga (justifica el uso de F1-Score):\n")
            f.write(df_rosales[col_plaga].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%").to_string() + "\n\n")

print(f"Reporte de texto generado y guardado como: {nombre_reporte}")

# 3. GENERACIÓN Y GUARDADO DE GRÁFICOS (.png)
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

if df_rosales is not None and not df_rosales.empty:
    col_plaga = 'Plant_Health_Status'
    col_humedad = 'Humidity' 
    
    if col_plaga in df_rosales.columns:
        # Gráfico de desbalanceo
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df_rosales, x=col_plaga, palette='Reds', hue=col_plaga, legend=False)
        plt.title('Desbalanceo de Clases: Días Sanos vs Plaga')
        plt.tight_layout()
        plt.savefig('desbalanceo_plagas_vitis.png')
        plt.close()
        print("Gráfico guardado: desbalanceo_plagas_vitis.png")
        
        # Gráfico de correlación Clima vs Plaga
        if col_humedad in df_rosales.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=col_plaga, y=col_humedad, data=df_rosales, palette='Blues', hue=col_plaga, legend=False)
            plt.title('Influencia de la Humedad en la Aparición de Plagas')
            plt.tight_layout()
            plt.savefig('humedad_vs_plaga_vitis.png')
            plt.close()
            print("Gráfico guardado: humedad_vs_plaga_vitis.png")

print("\n¡Proceso finalizado con éxito! Revisa la carpeta para ver tu reporte y tus gráficos.")