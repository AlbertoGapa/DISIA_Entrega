import pandas as pd
import requests
import time
import math

BASE_URL = "http://localhost:8000"

def inyectar_temporada_desde_csv():
    ruta_csv = 'data/temporada_simulada_2025.csv'
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print("No se encuentra el CSV")
        return

    print("Iniciando inyección de datos")
    
    for _, row in df.iterrows():
        fecha = row['fecha']
        
        payload_clima = {
            "fecha": fecha,
            "temp": row['temp'],
            "humedad": row['humedad'],
            "lluvia": row['lluvia']
        }
        r_clima = requests.post(f"{BASE_URL}/estacion/clima", json=payload_clima)
        print(f"[{fecha}] Clima -> T: {row['temp']}ºC | HTTP {r_clima.status_code}")

        payload_plagas = {
            "fecha": fecha,
            "Ambient_Temperature": row['temp'],
            "Humidity": row['humedad'],
            "Soil_Moisture": row['Soil_Moisture'],
            "Soil_Temperature": row['Soil_Temperature'],
            "Light_Intensity": row['Light_Intensity'],
            "Soil_pH": row['Soil_pH'],
            "Nitrogen_Level": row['Nitrogen_Level'],
            "Phosphorus_Level": row['Phosphorus_Level'],
            "Potassium_Level": row['Potassium_Level'],
            "Chlorophyll_Content": row['Chlorophyll_Content'],
            "Electrochemical_Signal": row['Electrochemical_Signal'],
            "salud_real": row['salud_real']
        }
        r_plagas = requests.post(f"{BASE_URL}/plagas/registrar_real", json=payload_plagas)

        if not math.isnan(row['brix']):
            payload_madurez = {
                "fecha": fecha,
                "variedad": "Syrah",
                "brix": row['brix'],
                "acidez": row['acidez'],
                "madurez_real": row['madurez_real']
            }
            r_mad = requests.post(f"{BASE_URL}/madurez/registrar_real", json=payload_madurez)
            
            if r_mad.status_code == 200:
                print(f"Madurez ({row['brix']} Brix) -> Registrada OK")

        # Retardo para no colapsar la API
        time.sleep(0.5) 

    print("Inyección completada")

if __name__ == "__main__":
    inyectar_temporada_desde_csv()