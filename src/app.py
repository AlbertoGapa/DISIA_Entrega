from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import requests
import os

app = FastAPI(
    title="API VITIS-IA Producción",
    description="Sistema inteligente de predicción agronómica y prescripción de riego para viñedos.",
    version="1.1.0"
)

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