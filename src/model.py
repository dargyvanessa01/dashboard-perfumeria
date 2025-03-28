from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np
import pandas as pd

# Modelo ARIMA mejorado con manejo de estacionalidad
def entrenar_modelo_arima(serie_temporal, diferencia=True):
    """
    Entrena modelo ARIMA con verificación automática de estacionariedad
    """
    if diferencia:
        serie = verificar_estacionariedad(serie_temporal)
    else:
        serie = serie_temporal.copy()
    
    model = ARIMA(serie, order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit()
    return model_fit

# Modelo Ridge con integración de parámetro alpha
def modelo_ridge(X, y, alpha=1.0):
    """
    Entrena modelo Ridge con regularización ajustable
    """
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

# Función de estacionariedad mejorada
def verificar_estacionariedad(serie):
    resultado = adfuller(serie)
    if resultado[1] > 0.05:
        return serie.diff().dropna()
    return serie

# Generación de características temporales
def crear_features(df):
    """
    Crea variables para análisis temporal:
    - Diferencia de ventas
    - Media móvil
    - Variables estacionales
    """
    df['ventas_diff'] = df['ventas'].diff()
    df['media_movil_3'] = df['ventas'].rolling(window=3).mean()
    df['trimestre'] = (df.index.month - 1) // 3 + 1
    return df.dropna()