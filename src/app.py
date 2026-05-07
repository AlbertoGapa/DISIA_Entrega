from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import requests
import os
import csv
from datetime import date
from typing import Optional
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

app = FastAPI(
    title="API VITIS-IA Producción",
    description="Sistema inteligente de predicción agronómica y prescripción de riego para viñedos.",
    version="1.2.0"
)

# Inicialización de métricas operativas para Prometheus
Instrumentator().instrument(app).expose(app)
# 1. Contador de predicciones realizadas
PREDICCIONES_TOTALES = Counter('vitis_ia_predictions_total', 'Total de predicciones realizadas', ['tipo_modelo'])
ERROR_MADUREZ_ACTUAL = Gauge('vitis_ia_madurez_error_pct', 'Error porcentual de la última validación de madurez')
DISTRIBUCION_MADUREZ = Histogram('vitis_ia_madurez_predicha_values', 'Distribución de los valores de madurez predichos', buckets=[0, 25, 50, 75, 90, 100])
PLAGAS_PREDICCIONES = Counter('vitis_ia_plagas_total', 'Total de predicciones de salud de rosales', ['estado_predicho'])
ERROR_PLAGAS_TOTAL = Counter('vitis_ia_plagas_errores_total', 'Total de veces que la IA falló comparado con la realidad')

# CARGA DE MODELOS 
try:
    modelo_plagas = joblib.load('models/modelo_plagas.pkl')
    modelo_maturity = joblib.load('models/modelo_maturity.pkl')
    columnas_maturity = joblib.load('models/columnas_maturity.pkl')
    print("✓ Modelos ML cargados correctamente desde el volumen.")
except Exception as e:
    print(f"Error al cargar modelos: {e}")
    modelo_plagas, modelo_maturity, columnas_maturity = None, None, None


# ESQUEMAS DE VALIDACIÓN
class DatosPlaga(BaseModel):
    Humidity: float = Field(..., ge=0, le=100)
    Soil_Moisture: float = Field(..., ge=0, le=100)
    Nitrogen_Level: float = Field(..., ge=0)
    Ambient_Temperature: float = Field(25.0)

class DatosMaturity(BaseModel):
    acidity: float = Field(..., gt=0)
    brix: float = Field(..., ge=0)
    variedad: str = Field(..., description="Xinomavro, Syrah o Sauvignon Blanc")

class DatosRiego(BaseModel):
    acidity: float
    brix: float
    variedad: str
    soil_moisture: float = Field(..., description="Humedad actual del suelo en %")
    codigo_municipio: str = Field("02039", description="Código AEMET de Higueruela, Albacete")

class DatosEstacion(BaseModel):
    fecha: Optional[date] = None
    temp: float
    humedad: float
    lluvia: float

class DatosMadurezReal(BaseModel):
    fecha: Optional[date] = None
    variedad: str
    brix: float
    acidez: float
    madurez_real: float

class DatosPlagaReal(BaseModel):
    fecha: Optional[date] = None
    Humidity: float
    Soil_Moisture: float
    Nitrogen_Level: float
    Ambient_Temperature: float
    salud_real: str


# FUNCIONES AUXILIARES 
def enviar_alerta_telegram(mensaje):
    """Envía notificaciones de alerta utilizando la API de Telegram."""
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[Aviso] Credenciales de Telegram no configuradas.")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": mensaje}, timeout=5)
    except Exception as e:
        print(f"Error en el servicio de alertas: {e}")

def guardar_en_csv(registro: dict, nombre_archivo: str):
    """Almacena registros de entrada en archivos CSV para el feedback loop."""
    os.makedirs('data', exist_ok=True)
    ruta = f'data/{nombre_archivo}'
    archivo_existe = os.path.isfile(ruta)
    
    with open(ruta, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=registro.keys())
        if not archivo_existe or os.stat(ruta).st_size == 0:
            writer.writeheader()
        writer.writerow(registro)


# ENDPOINTS
@app.get("/health")
def health():
    if modelo_plagas and modelo_maturity:
        return {"status": "healthy", "ready": True}
    return {"status": "unhealthy", "ready": False}

@app.post("/predict_plagas")
def predict_plagas(datos: DatosPlaga):
    if not modelo_plagas: raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    cols_modelo = modelo_plagas.feature_names_in_
    input_dict = {col: 50.0 for col in cols_modelo}
    input_dict.update(datos.dict())
    
    df_input = pd.DataFrame([input_dict])[cols_modelo]
    pred = int(modelo_plagas.predict(df_input)[0])
    
    mapeo = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}
    
    PLAGAS_PREDICCIONES.labels(estado_predicho=estado).inc()
    return {"status": "success", "result": mapeo[pred]}

@app.post("/predict_maturity")
def predict_maturity(datos: DatosMaturity):
    if not modelo_maturity or not columnas_maturity: raise HTTPException(status_code=503)
    
    input_dict = {col: 0 for col in columnas_maturity}
    input_dict['Total acidity (g/l TA)'] = datos.acidity
    input_dict['Brix'] = datos.brix
    
    col_var = f"Variedad_{datos.variedad}"
    if col_var in input_dict: input_dict[col_var] = 1
    
    df_input = pd.DataFrame([input_dict])[columnas_maturity]
    pred = float(modelo_maturity.predict(df_input)[0])
    
    PREDICCIONES_TOTALES.labels(tipo_modelo='madurez').inc() 
    DISTRIBUCION_MADUREZ.observe(pred) 
    return {
        "status": "success",
        "data": {
            "estimated_maturity_percentage": round(pred, 2),
            "units": "%"
        }
    }


# MOTOR DE REGLAS Y LLAMADA A AEMET
@app.post("/recommend_irrigation")
def recommend_irrigation(datos: DatosRiego):
    if not modelo_maturity: raise HTTPException(status_code=503)

    # INFERENCIA DEL MODELO ML (Maduración)
    input_dict = {col: 0 for col in columnas_maturity}
    input_dict['Total acidity (g/l TA)'] = datos.acidity
    input_dict['Brix'] = datos.brix
    col_var = f"Variedad_{datos.variedad}"
    if col_var in input_dict: input_dict[col_var] = 1
    
    df_input = pd.DataFrame([input_dict])[columnas_maturity]
    maduracion_predicha = float(modelo_maturity.predict(df_input)[0])

    # CONSULTA A LA API DE AEMET (Higueruela)
    prob_lluvia = 0
    temp_max = 25.0
    aemet_status = "ok"

    api_key_secreta = os.getenv("AEMET_API_KEY")

    if not api_key_secreta:
        aemet_status = "Aviso: API Key de AEMET no configurada en el servidor."
    else:
        try:
            url_aemet = f"https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/diaria/{datos.codigo_municipio}"
            params = {"api_key": api_key_secreta} 
            headers = {'cache-control': "no-cache"}

            # Primera llamada para obtener el enlace de descarga de datos
            res_link = requests.get(url_aemet, headers=headers, params=params, timeout=5)
            if res_link.status_code == 200:
                datos_url = res_link.json().get("datos")
                
                # Segunda llamada para bajar el JSON con el clima real
                res_datos = requests.get(datos_url, timeout=5)
                clima_json = res_datos.json()
                
                # Extraemos los datos de "hoy" 
                hoy = clima_json[0]['prediccion']['dia'][0]
                
                val_lluvia = hoy['probPrecipitacion'][0].get('value', '0')
                prob_lluvia = int(val_lluvia) if val_lluvia else 0
                temp_max = float(hoy['temperatura']['maxima'])
            else:
                aemet_status = f"Error auth/AEMET devuelve código {res_link.status_code}"
        except Exception as e:
            aemet_status = f"Fallo de conexión externa: {str(e)}"

    # MOTOR DE PRESCRIPCIÓN AGRONÓMICA
    recomendacion = ""
    motivo = ""

    # Regla 1: Estrés hídrico final (Prioridad Máxima)
    if maduracion_predicha > 90.0:
        recomendacion = "CORTAR RIEGO (ESTRÉS HÍDRICO)"
        motivo = f"Uva al {maduracion_predicha:.1f}% de maduración. Cortar el agua concentrará los azúcares finales para la vendimia."
    
    # Regla 2: Lluvia inminente según AEMET
    elif prob_lluvia >= 50:
        recomendacion = "NO REGAR (AHORRO)"
        motivo = f"AEMET pronostica un {prob_lluvia}% de probabilidad de lluvia en Higueruela. Aprovechamiento de agua meteórica."
    
    # Regla 3: Suelo seco y calor extremo
    elif datos.soil_moisture < 30.0 and temp_max > 30.0:
        recomendacion = "RIEGO DE APOYO URGENTE"
        motivo = f"Humedad del suelo crítica ({datos.soil_moisture}%) y calor extremo esperado en Higueruela ({temp_max}ºC)."
    
    # Regla 4: Suelo seco, pero clima suave
    elif datos.soil_moisture < 30.0:
        recomendacion = "RIEGO DE MANTENIMIENTO"
        motivo = f"Humedad del suelo baja ({datos.soil_moisture}%), reponer hasta el 50% de capacidad de campo."
    
    # Regla 5: Condiciones óptimas
    else:
        recomendacion = "NO REGAR"
        motivo = "El suelo de la finca tiene humedad suficiente y no hay riesgo térmico inminente."

    # RESPUESTA FINAL
    return {
        "status": "success",
        "system_data": {
            "ml_maturity_prediction_percentage": round(maduracion_predicha, 2),
            "aemet_connection_status": aemet_status,
            "location_monitored": "Higueruela, Albacete (02039)",
            "weather_forecast": {
                "rain_probability_pct": prob_lluvia,
                "max_temperature_celsius": temp_max
            }
        },
        "actionable_insight": {
            "irrigation_recommendation": recomendacion,
            "agronomic_reasoning": motivo
        }
    }


# ENDPOINTS MLOPS
@app.post("/estacion/clima", tags=["MLOps"])
async def registrar_clima_estacion(datos: DatosEstacion):
    fecha_final = datos.fecha if datos.fecha else date.today()
    registro = {
        "fecha": fecha_final.isoformat(),
        "temp": datos.temp,
        "humedad": datos.humedad,
        "lluvia": datos.lluvia,
    }
    guardar_en_csv(registro, 'buffer_reentrenamiento.csv')
    
    if datos.temp > 45.0:
        enviar_alerta_telegram(f"Alerta de Sistema: Temperatura extrema detectada ({datos.temp}ºC).")
    
    return {"status": "success", "data_logged": registro}

@app.post("/madurez/registrar_real", tags=["MLOps"])
async def registrar_madurez_real(datos: DatosMadurezReal):
    fecha_registro = datos.fecha if datos.fecha else date.today()
    error_ia = 0.0
    
    input_dict = {col: 0 for col in columnas_maturity}
    input_dict['Total acidity (g/l TA)'] = datos.acidez
    input_dict['Brix'] = datos.brix
    col_var = f"Variedad_{datos.variedad}"
    if col_var in input_dict: input_dict[col_var] = 1
    
    try:
        df_input = pd.DataFrame([input_dict])[columnas_maturity]
        pred_ia = modelo_maturity.predict(df_input)[0]
        error_ia = abs(pred_ia - datos.madurez_real)
        if error_ia > 15.0:
            enviar_alerta_telegram(f"Alerta de Modelo: Desviación crítica en madurez ({round(error_ia, 2)}%). Variedad: {datos.variedad}.")
    except Exception:
        pass

    registro = {
        "fecha": fecha_registro.isoformat(),
        "Variedad": datos.variedad,
        "Brix": datos.brix,
        "Total acidity (g/l TA)": datos.acidez,
        "Maturity percentage": datos.madurez_real,
        "error_ia": round(error_ia, 2)
    }
    guardar_en_csv(registro, 'reentrenamiento_madurez.csv')
    ERROR_MADUREZ_ACTUAL.set(error_ia) 
    return {"status": "success", "feedback": {"error_detectado": round(error_ia, 2)}}

@app.post("/plagas/registrar_real", tags=["MLOps"])
async def registrar_plaga_real(datos: DatosPlagaReal):
    fecha_registro = datos.fecha if datos.fecha else date.today()
    alerta_estado = "OK"
    
    try:
        if modelo_plagas is not None:
            input_df = pd.DataFrame([{
                "Humidity": datos.Humidity, "Soil_Moisture": datos.Soil_Moisture,
                "Nitrogen_Level": datos.Nitrogen_Level, "Ambient_Temperature": datos.Ambient_Temperature
            }])
            pred_num = modelo_plagas.predict(input_df)[0]
            mapa_inverso = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}
            pred_ia = mapa_inverso.get(pred_num, 'Unknown')
            
            if pred_ia != datos.salud_real:
                alerta_estado = f"Fallo: Predicho {pred_ia} vs Real {datos.salud_real}"
                enviar_alerta_telegram(f"Alerta de Modelo: Error en clasificación fitosanitaria. {alerta_estado}")
    except Exception:
        pass

    registro = {
        "fecha": fecha_registro.isoformat(),
        "Humidity": datos.Humidity, "Soil_Moisture": datos.Soil_Moisture,
        "Nitrogen_Level": datos.Nitrogen_Level, "Ambient_Temperature": datos.Ambient_Temperature,
        "salud_real": datos.salud_real
    }
    guardar_en_csv(registro, 'reentrenamiento_plagas.csv')
    
    if pred_ia != datos.salud_real:

        ERROR_PLAGAS_TOTAL.inc()
        enviar_alerta_telegram(f"🚨 Error de Clasificación: IA dijo {pred_ia} pero la realidad es {datos.salud_real}")
    
    return {"status": "success", "feedback": alerta_estado}

@app.post("/admin/recargar_modelos", tags=["MLOps"])
def recargar_modelos_en_caliente():
    global modelo_plagas, modelo_maturity, columnas_maturity
    try:
        modelo_plagas = joblib.load('models/modelo_plagas.pkl')
        modelo_maturity = joblib.load('models/modelo_maturity.pkl')
        columnas_maturity = joblib.load('models/columnas_maturity.pkl')
        return {"status": "success", "message": "Modelos actualizados correctamente en memoria."}
    except Exception as e:
        return {"status": "error", "message": f"Fallo al recargar modelos: {e}"}