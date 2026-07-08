from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="AirQ IA Microservice")

# 1. Carga de modelos
try:
    model = joblib.load('airq_universal_model.pkl')
    air_quality_model = joblib.load('air_quality_model.pkl')
    climate_model = joblib.load('climate_model.pkl')
    print("Modelos IA separados cargados y listos.")
except Exception as e:
    print(f"Error cargando los modelos: {e}")
    model = None
    air_quality_model = None
    climate_model = None

# 1.5 Endpoint de Health Check (Ping Ligero)
# Excluido de OpenAPI/Swagger (include_in_schema=False) para no consumir RAM al renderizar /docs
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/health", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/health/", methods=["GET", "HEAD"], include_in_schema=False)
def health_check():
    return {"status": "ok"}

# 2. Definir la estructura de comunicación con Spring Boot (Los 4 sensores)
class SensorData(BaseModel):
    co2: float
    pm25: float
    temp: float
    hum: float


class PredictionRequest(BaseModel):
    sensorId: str
    data: list[SensorData]  # Recibimos la ventana de los últimos minutos


class PredictionResponse(BaseModel):
    riskLevel: str
    aiActionTaken: str


# 3. El Endpoint de Inferencia
@app.post("/predict", response_model=PredictionResponse)
def predict_air_quality(request: PredictionRequest):
    if model is None or air_quality_model is None or climate_model is None:
        raise HTTPException(status_code=500, detail="Los modelos de IA no están disponibles.")

    if not request.data:
        raise HTTPException(status_code=400, detail="No se enviaron datos de los sensores.")

    # Tomamos el último registro (el más reciente) para la predicción inmediata
    latest_data = request.data[-1]

    # Formateamos exactamente igual a como entrenamos (co2, pm25, temp, hum)
    features = pd.DataFrame([{
        'co2': latest_data.co2,
        'pm25': latest_data.pm25,
        'temp': latest_data.temp,
        'hum': latest_data.hum
    }])

    # 4. Inferencia del Modelo Universal (deduce el riesgo general)
    riesgo = model.predict(features)[0]

    # 5. Inferencia de Modelos Separados (100% IA, cero sesgos)
    air_features = features[['co2', 'pm25']]
    climate_features = features[['temp', 'hum']]
    
    air_action_id = air_quality_model.predict(air_features)[0]
    climate_action_id = climate_model.predict(climate_features)[0]
    
    co2_val = air_features['co2'][0]
    pm25_val = air_features['pm25'][0]
    
    # Mapeo de Calidad de Aire (Controla Extractores, Filtros y Rejillas)
    air_dict = {
        0: "Calidad de aire óptima (Rejillas ABIERTAS)",
        1: f"Alerta Polvo (PM2.5: {pm25_val}). Filtro HEPA activo. Rejillas CERRADAS",
        2: f"Alerta CO2 ({co2_val} ppm). EXTRACTOR al máximo. Rejillas ABIERTAS",
        3: f"CRÍTICO: Polvo y Asfixia. EXTRACTOR al máximo, Filtro HEPA activo. Rejillas CERRADAS (Aislamiento)"
    }
    
    # Mapeo de Clima (Controla Aire Acondicionado)
    climate_dict = {
        0: "Confort térmico ideal",
        1: "Calor detectado. AC en modo COOL",
        2: "Humedad alta. AC en modo DRY",
        3: "Bochorno severo. AC en modo COOL y DRY"
    }
    
    # Fusión de los outputs de ambos modelos de IA
    action = f"{air_dict.get(air_action_id, 'Normal')} | {climate_dict.get(climate_action_id, 'Normal')}"

    return PredictionResponse(
        riskLevel=riesgo,
        aiActionTaken=action
    )


if __name__ == "__main__":
    import uvicorn
    import os

    # Render sets the PORT environment variable. Fallback to 5000 for local.
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)