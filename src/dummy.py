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
        
        # Enviar Clima
        payload_clima = {
            "fecha": fecha,
            "temp": row['temp'],
            "humedad": row['humedad'],
            "lluvia": row['lluvia']
        }
        r_clima = requests.post(f"{BASE_URL}/estacion/clima", json=payload_clima)
        print(f"[{fecha}] Clima -> T: {row['temp']}ºC | HTTP {r_clima.status_code}")

        # Enviar Plagas
        payload_plagas = {
            "fecha": fecha,
            "Humidity": row['humedad'],
            "Soil_Moisture": round(row['humedad'] * 0.8, 2), # Aproximación
            "Nitrogen_Level": 40.0,
            "Ambient_Temperature": row['temp'],
            "salud_real": row['salud_real']
        }
        r_plagas = requests.post(f"{BASE_URL}/plagas/registrar_real", json=payload_plagas)

        # Enviar Madurez
        if not math.isnan(row['brix']):
            payload_madurez = {
                "fecha": fecha,
                "variedad": "Syrah",
                "brix": row['brix'],
                "acidez": row['acidez'],
                "madurez_real": row['madurez_real']
            }
            r_mad = requests.post(f"{BASE_URL}/madurez/registrar_real", json=payload_madurez)
            
            # Ver si el modelo se ha equivocado 
            if r_mad.status_code == 200:
                alerta = r_mad.json().get("feedback_metrics", {}).get("alerta", "")
                estado = f"{alerta}" if "Crítico" in alerta else "OK"
                print(f"Madurez ({row['brix']} Brix) -> {estado}")

        # Retardo para que las métricas de Prometheus / FastAPI sean realistas
        time.sleep(0.15) 

    print("Inyección completada")

if __name__ == "__main__":
    inyectar_temporada_desde_csv()