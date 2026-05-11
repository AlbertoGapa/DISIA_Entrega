from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
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

PREDICCIONES_TOTALES = Counter('vitis_ia_predictions_total', 'Total de predicciones realizadas', ['tipo_modelo'])
ERROR_MADUREZ_ACTUAL = Gauge('vitis_ia_madurez_error_pct', 'Error porcentual de la última validación de madurez')
DISTRIBUCION_MADUREZ = Histogram('vitis_ia_madurez_predicha_values', 'Distribución de los valores de madurez predichos', buckets=[0, 25, 50, 75, 90, 100])
PLAGAS_PREDICCIONES = Gauge('vitis_ia_plagas_predicciones', 'Total de predicciones de salud de rosales')
SALUD_REAL_VALOR = Gauge('vitis_ia_plagas_real', 'Último estado de salud real por el laboratorio')
ERROR_PLAGAS_TOTAL = Counter('vitis_ia_plagas_errores_total', 'Total de veces que la IA falló comparado con la realidad')
AEMET_LLUVIA_PROB = Gauge('vitis_ia_aemet_lluvia_prob', 'Probabilidad de lluvia actual obtenida de AEMET')
MADUREZ_PREDICHA_ACTUAL = Gauge('vitis_ia_madurez_predicha', 'Último porcentaje de madurez predicho por el modelo')
MADUREZ_REAL_ACTUAL = Gauge('vitis_ia_madurez_real', 'Último porcentaje de madurez real por el laboratorio')
ESTACION_LLUVIA = Gauge('vitis_ia_estacion_lluvia_mm', 'Lluvia real registrada por la estación en el viñedo (mm)')
AEMET_LLUVIA_PREVISTA_MM = Gauge('vitis_ia_aemet_lluvia_prevista_mm', 'Cantidad de lluvia prevista por AEMET (mm)')
AEMET_TEMP_MAX = Gauge('vitis_ia_aemet_temp_max', 'Temperatura máxima prevista por AEMET')
ESTACION_TEMP = Gauge('vitis_ia_estacion_temp', 'Temperatura real medida en la finca')
ESTACION_HUMEDAD = Gauge('vitis_ia_estacion_humedad', 'Humedad real medida en la finca')


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
class DatosPrediccionPlaga(BaseModel):
    Soil_Moisture: float = Field(default=0.0)
    Ambient_Temperature: float = Field(default=0.0)
    Soil_Temperature: float = Field(default=22.0)
    Humidity: float = Field(default=0.0)
    Light_Intensity: float = Field(default=4000.0)
    Soil_pH: float = Field(default=6.5)
    Nitrogen_Level: float = Field(default=0.0)
    Phosphorus_Level: float = Field(default=20.0)
    Potassium_Level: float = Field(default=30.0)
    Chlorophyll_Content: float = Field(default=45.0)
    Electrochemical_Signal: float = Field(default=15.0)

class DatosMaturity(BaseModel):
    acidity: float = Field(..., gt=0)
    brix: float = Field(..., ge=0)
    variedad: str = Field("Syrah", description="Xinomavro, Syrah o Sauvignon Blanc")

class DatosRiego(BaseModel):
    acidity: float
    brix: float
    variedad: str = Field("Syrah")
    soil_moisture: float = Field(..., description="Humedad actual del suelo en %")
    codigo_municipio: str = Field("02039", description="Código AEMET de Higueruela, Albacete")

class DatosEstacion(BaseModel):
    fecha: Optional[date] = None
    temp: float
    humedad: float
    lluvia: float = Field(0.0, description="Litros por metro cuadrado (mm) caídos")

class DatosMadurezReal(BaseModel):
    fecha: Optional[date] = None
    variedad: str
    brix: float
    acidez: float
    madurez_real: float

class DatosPlagaReal(BaseModel):
    salud_real: str = Field(..., description="Healthy, Moderate Stress o High Stress")
    fecha: Optional[date] = None
    Soil_Moisture: float = Field(default=0.0)
    Ambient_Temperature: float = Field(default=0.0)
    Soil_Temperature: float = Field(default=22.0)
    Humidity: float = Field(default=0.0)
    Light_Intensity: float = Field(default=4000.0)
    Soil_pH: float = Field(default=6.5)
    Nitrogen_Level: float = Field(default=0.0)
    Phosphorus_Level: float = Field(default=20.0)
    Potassium_Level: float = Field(default=30.0)
    Chlorophyll_Content: float = Field(default=45.0)
    Electrochemical_Signal: float = Field(default=15.0)


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

@app.post("/predict_plaga", tags=["Inferencia"])
def predict_plaga(datos: DatosPrediccionPlaga):
    if modelo_plagas is None:
        raise HTTPException(status_code=503, detail="Modelo de plagas no cargado")

    try:

        cols_modelo = list(modelo_plagas.feature_names_in_)
        datos_dict = datos.model_dump()
        input_dict = {col: datos_dict.get(col, 0.0) for col in cols_modelo}
        df_input = pd.DataFrame([input_dict], columns=cols_modelo)
        pred_num = int(modelo_plagas.predict(df_input)[0])
        
        mapa_inverso = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}
        resultado = mapa_inverso.get(pred_num, 'Unknown')
        PREDICCIONES_TOTALES.labels(tipo_modelo='plagas').inc()
        PLAGAS_PREDICCIONES.set(mapa_inverso.get(resultado, -1))

        return {
            "status": "success",
            "prediction": resultado,
            "prediction_code": pred_num,
            "monitored_sensors": len(input_dict)
        }

    except Exception as e:
        print(f"🚨 Error en predict_plaga: {e}")
        raise HTTPException(status_code=500, detail=f"Error en el motor de inferencia: {str(e)}")

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
    MADUREZ_PREDICHA_ACTUAL.set(pred)
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
        prob_lluvia = 0.0
    mm_lluvia_prevista = 0.0
    estado_aemet = "OK"
    
    try:
        api_key = os.getenv("AEMET_API_KEY")
        url = f"https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/horaria/{datos.codigo_municipio}/?api_key={api_key}"
        
        respuesta = requests.get(url, timeout=5)
        
        if respuesta.status_code == 200:
            datos_url = respuesta.json().get("datos")
            if datos_url:
                respuesta_datos = requests.get(datos_url, timeout=5)
                datos_tiempo = respuesta_datos.json()
                
                # Obtenemos los datos del día de HOY
                dia_hoy = datos_tiempo[0]['prediccion']['dia'][0]
                
                # TOTAL DE LLUVIA HOY
                for precipitacion in dia_hoy.get('precipitacion', []):
                    mm_str = precipitacion.get('value')
                    # 'Ip' es Inapreciable
                    if mm_str and mm_str != 'Ip':
                        mm_lluvia_prevista += float(mm_str)
                        
                # PROBABILIDAD MÁXIMA DE HOY
                for prob in dia_hoy.get('probPrecipitacion', []):
                    prob_str = prob.get('value')
                    if prob_str:
                        prob_lluvia = max(prob_lluvia, float(prob_str))

        elif respuesta.status_code == 429:
            estado_aemet = "Límite AEMET 429. Usando fallback (0mm, 0%)."
        else:
            estado_aemet = f"Error AEMET: {respuesta.status_code}"
            
    except Exception as e:
        estado_aemet = f"Fallo conexión AEMET: {str(e)}"

    AEMET_LLUVIA_PROB.set(prob_lluvia)
    AEMET_LLUVIA_PREVISTA_MM.set(mm_lluvia_prevista)
    AEMET_TEMP_MAX.set(temp_max)

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
    
    ESTACION_LLUVIA.set(datos.lluvia)
    ESTACION_TEMP.set(datos.temp)
    ESTACION_HUMEDAD.set(datos.humedad)
    
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
        error_ia = round(abs(pred_ia - datos.madurez_real), 2)
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
    MADUREZ_REAL_ACTUAL.set(datos.madurez_real) 
    ERROR_MADUREZ_ACTUAL.set(error_ia) 
    MADUREZ_PREDICHA_ACTUAL.set(pred_ia)
    
    return {"status": "success", "feedback": {"error_detectado": round(error_ia, 2)}}

@app.post("/plagas/registrar_real", tags=["MLOps"])
def registrar_plaga_real(datos: DatosPlagaReal):
    fecha_registro = datos.fecha if datos.fecha else date.today()
    alerta_estado = "OK"
    pred_ia = "Unknown"
    mapeo_grafana = {'Healthy': 0, 'Moderate Stress': 1, 'High Stress': 2}
    
    try:
        if modelo_plagas is not None:

            cols_modelo = list(modelo_plagas.feature_names_in_)
            datos_dict = datos.model_dump()
            input_dict = {col: datos_dict.get(col, 0.0) for col in cols_modelo}
            df_input = pd.DataFrame([input_dict], columns=cols_modelo)
            
            # Predicción
            pred_num = int(modelo_plagas.predict(df_input)[0])
            mapa_inverso = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}
            pred_ia = mapa_inverso.get(pred_num, 'Unknown')
            
            PLAGAS_PREDICCIONES.set(mapeo_grafana.get(pred_ia, -1))

    except Exception as e:
        print(f"Error crítico en inferencia plagas: {e}")
        alerta_estado = f"Error técnico: {str(e)}"

    # Persistencia en CSV
    registro = {
        "Timestamp": fecha_registro.isoformat(),
        "Plant_ID": "Feedback_IoT_01", 
        "Soil_Moisture": datos.Soil_Moisture,
        "Ambient_Temperature": datos.Ambient_Temperature,
        "Soil_Temperature": datos.Soil_Temperature,
        "Humidity": datos.Humidity,
        "Light_Intensity": datos.Light_Intensity,
        "Soil_pH": datos.Soil_pH,
        "Nitrogen_Level": datos.Nitrogen_Level,
        "Phosphorus_Level": datos.Phosphorus_Level,
        "Potassium_Level": datos.Potassium_Level,
        "Chlorophyll_Content": datos.Chlorophyll_Content,
        "Electrochemical_Signal": datos.Electrochemical_Signal,
        "Plant_Health_Status": datos.salud_real
    }
    
    guardar_en_csv(registro, 'reentrenamiento_plagas.csv')
    
    # Alertas y Métricas
    if pred_ia != "Unknown" and pred_ia != datos.salud_real:

        ERROR_PLAGAS_TOTAL.inc() 
        alerta_estado = f"Fallo IA: Predicho {pred_ia} vs Real {datos.salud_real}"
        enviar_alerta_telegram(f"Error de Clasificación: IA dijo {pred_ia} pero la realidad es {datos.salud_real}")
    
    
    SALUD_REAL_VALOR.set(mapeo_grafana.get(datos.salud_real, -1))

    return {"status": "success", "feedback": alerta_estado, "ia_prediction": pred_ia}

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

# Alerta Excepciones 
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    mensaje_error = f"ALERTA OPERATIVA: Error 500 en {request.url.path}. Detalle: {str(exc)}"
    enviar_alerta_telegram(mensaje_error) 
    
    return JSONResponse(
        status_code=500,
        content={"message": "Error interno del servidor", "detail": str(exc)},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    mensaje = f"ALERTA OPERATIVA: Datos mal formados en {request.url.path}. Errores: {exc.errors()}"
    enviar_alerta_telegram(mensaje) 
    return JSONResponse(
        status_code=422,
        content={"message": "Error de validación de datos", "detail": exc.errors()},
    )