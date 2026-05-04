import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (mean_squared_error, f1_score, recall_score, 
                             confusion_matrix, ConfusionMatrixDisplay)
import joblib


warnings.filterwarnings("ignore")

print("Iniciando Pipeline de Modelado para VITIS-IA \n")

try:
    df_xino = pd.read_excel('grape-maturity-dataset.xlsx', sheet_name='Xinomavro')
    df_syrah = pd.read_excel('grape-maturity-dataset.xlsx', sheet_name='Syrah')
    df_sauv = pd.read_excel('grape-maturity-dataset.xlsx', sheet_name='Sauvignon Blanc')
    
    df_xino['Variedad'] = 'Xinomavro'
    df_syrah['Variedad'] = 'Syrah'
    df_sauv['Variedad'] = 'Sauvignon Blanc'
    
    df_uvas = pd.concat([df_xino, df_syrah, df_sauv], ignore_index=True)
    print("✓ Dataset de Uvas cargado correctamente.")
    
except FileNotFoundError:
    print("ERROR: No se encontró 'grape-maturity-dataset.xlsx'.")
    exit()

# Limpieza y selección de features para Uvas
df_uvas = df_uvas.dropna(subset=['Maturity percentage'])
# Usamos acidez, brix y variedad para predecir la maduración
features_uvas = ['Total acidity (g/l TA)', 'Brix', 'Variedad']
X_reg = pd.get_dummies(df_uvas[features_uvas], drop_first=True)
X_reg = X_reg.fillna(X_reg.mean()) 
y_reg = df_uvas['Maturity percentage']

try:
    df_rosales = pd.read_csv('plant_health_data.csv')
    print("✓ Dataset de Rosales cargado correctamente.\n")
except FileNotFoundError:
    print("ERROR: No se encontró 'plant_health_data.csv'.")
    exit()

col_target_plaga = 'Plant_Health_Status' 
df_rosales = df_rosales.dropna(subset=[col_target_plaga])
cols_to_drop = ['Timestamp', 'Plant_ID']
df_rosales = df_rosales.drop(columns=[col for col in cols_to_drop if col in df_rosales.columns])

df_rosales[col_target_plaga] = df_rosales[col_target_plaga].str.strip().str.title()
mapeo_salud = {'Healthy': 0, 'Moderate Stress': 1, 'High Stress': 2}
df_rosales[col_target_plaga] = df_rosales[col_target_plaga].map(mapeo_salud)

df_rosales = df_rosales.dropna(subset=[col_target_plaga])
y_clf = df_rosales[col_target_plaga].astype(int)

X_clf = df_rosales.drop(columns=[col_target_plaga]).select_dtypes(include=[np.number])
X_clf = X_clf.fillna(X_clf.mean())


# MODELADO DE PLAGAS (CLASIFICACIÓN)

print("--- ENTRENANDO MODELOS DE PLAGAS (CLASIFICACIÓN) ---")
# Stratify asegura que la proporción de plagas se mantenga igual en train y test
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Baseline: Árbol de Decisión (Explicable)
clf_base = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
clf_base.fit(X_train_c, y_train_c)
y_pred_base = clf_base.predict(X_test_c)

print("Resultados Baseline (Árbol de Decisión - Interpretable):")
print(f"  Recall (Sensibilidad): {recall_score(y_test_c, y_pred_base, average='macro'):.4f}")
print(f"  F1-Score: {f1_score(y_test_c, y_pred_base, average='macro'):.4f}")

# GridSearch: Random Forest (Alta Precisión)
rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid_clf = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10]
}

# Maximizamos el F1-Score 
grid_clf = GridSearchCV(rf_clf, param_grid_clf, cv=5, scoring='f1_macro', n_jobs=-1)
grid_clf.fit(X_train_c, y_train_c)

mejor_clf = grid_clf.best_estimator_
y_pred_mej = mejor_clf.predict(X_test_c)

print("\nResultados Avanzados (Random Forest Optimizado vía GridSearch):")
print(f"  Mejores Parámetros: {grid_clf.best_params_}")
print(f"  Recall (Sensibilidad): {recall_score(y_test_c, y_pred_mej, average='macro'):.4f}")
print(f"  F1-Score: {f1_score(y_test_c, y_pred_mej, average='macro'):.4f}")

# Matriz de Confusión 
cm = confusion_matrix(y_test_c, y_pred_mej)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión - Estado del Rosal\n(Modelo Optimizado)')
plt.ylabel('Estado Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
plt.savefig('matriz_confusion_plagas.png')
plt.close()
print("-> Gráfico guardado: matriz_confusion_plagas.png\n")


# MODELADO DE MADURACIÓN (REGRESIÓN)

print("--- ENTRENANDO MODELOS DE MADURACIÓN (REGRESIÓN DE MATURITY) ---")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Baseline: Regresión Lineal
reg_base = LinearRegression()
reg_base.fit(X_train_r, y_train_r)
y_pred_r_base = reg_base.predict(X_test_r)

print("Resultados Baseline (Regresión Lineal):")
rmse_base = np.sqrt(mean_squared_error(y_test_r, y_pred_r_base))
print(f"  RMSE: {rmse_base:.4f} % Maduración")

# GridSearch: Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)
param_grid_reg = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None]
}

# Optimizamos el error cuadrático medio negativo
grid_reg = GridSearchCV(rf_reg, param_grid_reg, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_reg.fit(X_train_r, y_train_r)

mejor_reg = grid_reg.best_estimator_
y_pred_r_mej = mejor_reg.predict(X_test_r)
rmse_mej = np.sqrt(mean_squared_error(y_test_r, y_pred_r_mej))

print("\nResultados Avanzados (Random Forest Optimizado vía GridSearch):")
print(f"  Mejores Parámetros: {grid_reg.best_params_}")
print(f"  RMSE: {rmse_mej:.4f} % Maduración")

# Gráfico de Dispersión (Real vs Predicho)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_r, y_pred_r_mej, alpha=0.6, color='purple')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2) # Línea de perfección
plt.title('Predicción de Maduración: Real vs. Predicho')
plt.xlabel('Porcentaje de Maduración Real (%)')
plt.ylabel('Porcentaje de Maduración Predicho (%)')
plt.tight_layout()
plt.savefig('dispersion_maturity.png')
plt.close()
print("-> Gráfico guardado: dispersion_maturity.png")

print("\n--- GENERANDO EXPLICABILIDAD (SHAP) PARA EL HITO 3 ---")
# Generamos la explicación para el modelo de plagas (Random Forest)
explainer = shap.TreeExplainer(mejor_clf)
shap_values = explainer.shap_values(X_test_c)

# SHAP para clasificación multiclase devuelve una lista de matrices (una por clase).
# Queremos explicar la clase 2 ("High Stress" o Plaga Severa).
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values[:, :, 2], X_test_c, show=False)
plt.title('Importancia de Variables (SHAP) - Predicción de Estrés Alto', pad=20)
plt.tight_layout()
plt.savefig('shap_explicabilidad_plagas.png')
plt.close()
print("-> Gráfico guardado: shap_explicabilidad_plagas.png")

print("\n¡Pipeline de Machine Learning finalizado con éxito!")

# Guardar el modelo de clasificación de plagas
joblib.dump(mejor_clf, 'modelo_plagas.pkl')
print("Modelo de plagas guardado como modelo_plagas.pkl")

# Guardar el modelo de regresión de maduración
joblib.dump(mejor_reg, 'modelo_maturity.pkl')
joblib.dump(list(X_reg.columns), 'columnas_maturity.pkl')
print("Modelo de regresión guardado como modelo_maturity.pkl y columnas_maturity.pkl")