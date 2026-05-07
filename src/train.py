import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error

warnings.filterwarnings("ignore")

print("Iniciando Pipeline de Entrenamiento para VITIS-IA\n")

# Crear carpeta para modelos
os.makedirs('models', exist_ok=True)

try:
    # CARGA DE MADUREZ (Excel + Feedback)
    xls = pd.ExcelFile('data/grape-maturity-dataset.xlsx')
    lista_dfs = []
    for nombre_hoja in xls.sheet_names:
        df_hoja = pd.read_excel('data/grape-maturity-dataset.xlsx', sheet_name=nombre_hoja)
        df_hoja['Variedad'] = nombre_hoja
        lista_dfs.append(df_hoja)
    
    df_uvas = pd.concat(lista_dfs, ignore_index=True)
    df_uvas = df_uvas.dropna(subset=['Maturity percentage', 'Brix', 'Total acidity (g/l TA)'])
    
    if os.path.exists('data/reentrenamiento_madurez.csv'):
        df_feedback_m = pd.read_csv('data/reentrenamiento_madurez.csv')
        df_uvas = pd.concat([df_uvas, df_feedback_m], ignore_index=True)
        print(f"Feedback loop aplicado: {len(df_feedback_m)} registros nuevos de maduración.")

    # CARGA DE PLAGAS (CSV + Feedback)
    df_rosales = pd.read_csv('data/plant_health_data.csv')
    if os.path.exists('data/reentrenamiento_plagas.csv'):
        df_feedback_p = pd.read_csv('data/reentrenamiento_plagas.csv')
        df_feedback_p = df_feedback_p.rename(columns={'salud_real': 'Plant_Health_Status'})
        df_rosales = pd.concat([df_rosales, df_feedback_p], ignore_index=True)
        print(f"Feedback loop aplicado: {len(df_feedback_p)} registros nuevos fitosanitarios.")

except Exception as e:
    print(f"Error crítico cargando datos: {e}")
    exit()

# ENTRENAMIENTO MADURACIÓN (REGRESIÓN)
features_uvas = ['Total acidity (g/l TA)', 'Brix', 'Variedad']
X_reg = pd.get_dummies(df_uvas[features_uvas], drop_first=True)
y_reg = df_uvas['Maturity percentage']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train_r, y_train_r)

joblib.dump(rf_reg, 'models/modelo_maturity.pkl')
joblib.dump(list(X_reg.columns), 'models/columnas_maturity.pkl')
print("Modelo de Maduración guardado")

# ENTRENAMIENTO SALUD (CLASIFICACIÓN)
df_rosales['Plant_Health_Status'] = df_rosales['Plant_Health_Status'].str.strip().str.title()
mapeo_salud = {'Healthy': 0, 'Moderate Stress': 1, 'High Stress': 2}
df_rosales['target'] = df_rosales['Plant_Health_Status'].map(mapeo_salud)
df_rosales = df_rosales.dropna(subset=['target'])

X_clf = df_rosales.drop(columns=['Plant_Health_Status', 'Timestamp', 'Plant_ID', 'target'], errors='ignore').select_dtypes(include=[np.number])
y_clf = df_rosales['target'].astype(int)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf_clf.fit(X_train_c, y_train_c)

joblib.dump(rf_clf, 'models/modelo_plagas.pkl')
print("Modelo de Plagas guardado")

# CÁLCULO DE MÉTRICAS 
rmse = np.sqrt(mean_squared_error(y_test_r, rf_reg.predict(X_test_r)))
f1 = f1_score(y_test_c, rf_clf.predict(X_test_c), average='macro')

with open('models/resultado_test.txt', 'w') as f:
    f.write(f"RMSE Maduración: {rmse:.4f}\nF1-Score Salud (Macro): {f1:.4f}")

print(f"\n✓ Resultados guardados en models/resultado_test.txt")
print(f"F1-Score Plagas: {f1:.4f} | RMSE Maduración: {rmse:.4f}")