import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("=== INICIANDO ENTRENAMIENTO DE IA SEPARADA ===")

np.random.seed(42)
n_samples = 25000

# Distribuciones
co2 = np.random.uniform(400, 1500, n_samples)
pm25 = np.random.uniform(5, 60, n_samples)
temp = np.random.uniform(15, 35, n_samples)
hum = np.random.uniform(30, 95, n_samples)

# --- 1. MODELO DE CALIDAD DEL AIRE (TOXICIDAD) ---
df_air = pd.DataFrame({'co2': co2, 'pm25': pm25})

def get_air_action(row):
    co2_alto = row['co2'] >= 900
    pm25_alto = row['pm25'] >= 35.0
    
    if pm25_alto and co2_alto: 
        return 3 # CRITICAL: Extractor + HEPA + Cerrar Rejillas (Aislamiento)
    elif co2_alto: 
        return 2 # ALERT: Extractor + Abrir Rejillas
    elif pm25_alto: 
        return 1 # WARNING: Filtro HEPA + Cerrar Rejillas
    else: 
        return 0 # NORMAL

df_air['action'] = df_air.apply(get_air_action, axis=1)

X_air = df_air[['co2', 'pm25']]
y_air = df_air['action']

air_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
air_model.fit(X_air, y_air)
joblib.dump(air_model, 'air_quality_model.pkl')
print("Modelo de Calidad del Aire guardado como 'air_quality_model.pkl'")

# --- 2. MODELO DE CLIMA (CONFORT) ---
df_climate = pd.DataFrame({'temp': temp, 'hum': hum})

def get_climate_action(row):
    temp_alta = row['temp'] >= 28.0
    hum_alta = row['hum'] >= 70.0
    
    if temp_alta and hum_alta: 
        return 3 # Bochorno (COOL + DRY)
    elif temp_alta: 
        return 1 # Calor (COOL)
    elif hum_alta: 
        return 2 # Humedad (DRY)
    else: 
        return 0 # Ideal

df_climate['action'] = df_climate.apply(get_climate_action, axis=1)

X_clim = df_climate[['temp', 'hum']]
y_clim = df_climate['action']

clim_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clim_model.fit(X_clim, y_clim)
joblib.dump(clim_model, 'climate_model.pkl')
print("Modelo de Clima guardado como 'climate_model.pkl'")
