import requests
import time

BASE_URL = "http://localhost:8000"

def disparar_anomalia_operativa():
    print("\n Generando Anomalía Operativa (Error 500)")
    payload_corrupto = {"campo_inventado": "rompera api"}
    try:
        response = requests.post(f"{BASE_URL}/plagas/registrar_real", json=payload_corrupto)
        print(f"Resultado: HTTP {response.status_code} - Alerta de error enviada a Telegram")
    except Exception as e:
        print(f"API caída o error de conexión: {e}")

def disparar_anomalia_de_modelo():
    print("\nGenerando Anomalía de Datos")
    # Enviamos valores imposibles
    payload_drift = {
        "Ambient_Temperature": 200.0, 
        "Humidity": -50.0,
        "Soil_Moisture": 500.0,
        "salud_real": "Healthy" 
    }
    response = requests.post(f"{BASE_URL}/plagas/registrar_real", json=payload_drift)
    print(f"Resultado: IA predijo {response.json().get('ia_prediction')} | Feedback: {response.json().get('feedback')}")
    print("Alerta de precisión enviada a Telegram")

def disparar_anomalia_madurez():
    print("\nGenerando Fallo de Madurez")

    payload = {
        "fecha": "2025-10-10",
        "variedad": "Syrah",
        "brix": 25.0,       
        "acidez": 3.0,       
        "madurez_real": 10.0 
    }
    response = requests.post(f"{BASE_URL}/madurez/registrar_real", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"IA predijo: {data.get('ia_prediction')}% | Real: 10.0%")
        print(f"Feedback: {data.get('feedback_metrics', {}).get('alerta')}")
    else:
        print(f"Error en la petición: {response.status_code}")

if __name__ == "__main__":
    print("Inicio ataque")
    
    # Alerta Operativa
    disparar_anomalia_operativa()
    
    time.sleep(2)
    
    # Alerta de Modelo
    disparar_anomalia_de_modelo()
    disparar_anomalia_madurez()
    
    print("Fin ataque")
    
    print("\nSimulación completada revisa Telegram y Grafana")