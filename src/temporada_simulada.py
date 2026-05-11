import pandas as pd
import numpy as np
from datetime import timedelta, date
import os

def crear_csv_temporada():
    print("Generando Dataset de temporada (Mayo - Octubre)")
    fecha_inicio = date(2025, 5, 1)
    fecha_fin = date(2025, 10, 7)
    dias_totales = (fecha_fin - fecha_inicio).days + 1
    
    fechas = [fecha_inicio + timedelta(days=i) for i in range(dias_totales)]
    datos = []
    
    for fecha in fechas:
        mes = fecha.month
        dia_año = fecha.timetuple().tm_yday
        
        # Lógica del Clima (Higueruela, Albacete)
        if mes == 5:    temp_base, hum_base, prob_lluvia = 18, 60, 0.15
        elif mes == 6:  temp_base, hum_base, prob_lluvia = 26, 40, 0.05
        elif mes == 7:  temp_base, hum_base, prob_lluvia = 34, 20, 0.01 # Ola de calor
        elif mes == 8:  temp_base, hum_base, prob_lluvia = 32, 25, 0.05
        elif mes == 9:  temp_base, hum_base, prob_lluvia = 24, 45, 0.10
        else:           temp_base, hum_base, prob_lluvia = 18, 55, 0.15 # Octubre
        
        # Añadir ruido realista a las variables climáticas básicas
        temp = temp_base + np.random.normal(0, 2.5)
        hum = hum_base + np.random.normal(0, 5.0)
        lluvia = np.random.choice([0.0, np.random.uniform(2.0, 15.0)], p=[1-prob_lluvia, prob_lluvia])
        soil_moisture = max(5.0, hum * 0.8 + (lluvia * 2)) # La lluvia aumenta la humedad del suelo
        
        # Lógica de Salud
        if temp > 35.0 and hum < 25.0: salud = "High Stress" # Estrés hídrico
        elif hum > 65.0 and temp > 20.0: salud = "High Stress" # Hongos
        elif temp > 28.0: salud = "Moderate Stress"
        else: salud = "Healthy"
        
        soil_temp = round(temp - np.random.uniform(1.0, 3.0), 2)
        light = round(np.random.normal(4500 if mes < 9 else 3500, 500), 2)
        ph = round(np.random.normal(6.5, 0.2), 2)
        nitrogen = round(np.random.normal(45.0, 5.0), 2)
        phosphorus = round(np.random.normal(25.0, 3.0), 2)
        potassium = round(np.random.normal(35.0, 4.0), 2)
        
        # Sensores de la planta (Reaccionan a la salud real)
        if salud == "Healthy":
            chlorophyll = round(np.random.normal(48.0, 3.0), 2)
            electro = round(np.random.normal(16.0, 1.5), 2)
        elif salud == "Moderate Stress":
            chlorophyll = round(np.random.normal(35.0, 4.0), 2)
            electro = round(np.random.normal(12.0, 2.0), 2)
        else: # High Stress
            chlorophyll = round(np.random.normal(22.0, 5.0), 2)
            electro = round(np.random.normal(8.0, 2.0), 2)
            
        # Lógica de Madurez a partir de Agosto
        brix, acidez, madurez = None, None, None
        if mes >= 8:
            progreso = (dia_año - 213) / (280 - 213) 
            brix = round(10.0 + (progreso * 15.0) + np.random.normal(0, 0.5), 2)
            acidez = round(14.0 - (progreso * 10.0) + np.random.normal(0, 0.3), 2)
            madurez = round(30.0 + (progreso * 70.0) + np.random.normal(0, 2.0), 2)
            madurez = min(100.0, max(0.0, madurez)) 

        datos.append({
            "fecha": fecha.isoformat(),
            "temp": round(temp, 2),
            "humedad": round(max(5.0, hum), 2),
            "lluvia": round(lluvia, 2),
            "salud_real": salud,
            "brix": brix,
            "acidez": acidez,
            "madurez_real": madurez,
            "Soil_Moisture": round(soil_moisture, 2),
            "Soil_Temperature": soil_temp,
            "Light_Intensity": light,
            "Soil_pH": ph,
            "Nitrogen_Level": nitrogen,
            "Phosphorus_Level": phosphorus,
            "Potassium_Level": potassium,
            "Chlorophyll_Content": chlorophyll,
            "Electrochemical_Signal": electro
        })
        
    df = pd.DataFrame(datos)
    os.makedirs('data', exist_ok=True)
    ruta_csv = 'data/temporada_simulada_2025.csv'
    df.to_csv(ruta_csv, index=False)
    print(f"CSV generado con éxito: {ruta_csv} ({len(df)} días registrados con telemetría completa).")

if __name__ == "__main__":
    crear_csv_temporada()