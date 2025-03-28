from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# Modelo ARIMA para pronóstico de ventas
def entrenar_modelo_arima(serie_temporal):
    """
    Entrena un modelo ARIMA (1,1,1) para la serie temporal.
    :param serie_temporal: Serie temporal de ventas
    :return: Modelo ARIMA entrenado
    """
    model = ARIMA(serie_temporal, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    return model_fit

# Regularización con Ridge Regression
def modelo_ridge(X, y):
    """
    Entrena un modelo Ridge con validación cruzada en series temporales.
    :param X: Variables predictoras
    :param y: Variable objetivo
    :return: Mejor modelo Ridge ajustado
    """
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    tscv = TimeSeriesSplit(n_splits=2)  # Reducir divisiones para datos pequeños
    grid = GridSearchCV(Ridge(), param_grid, cv=tscv)
    grid.fit(X, y)
    return grid.best_estimator_

# Verificar estacionariedad de la serie temporal
def verificar_estacionariedad(serie):
    resultado = adfuller(serie)
    print(f"ADF Statistic: {resultado[0]}")
    print(f"p-value: {resultado[1]}")
    if resultado[1] > 0.05:
        print("La serie no es estacionaria. Aplicando diferenciación...")
        return serie.diff().dropna()
    else:
        print("La serie es estacionaria.")
        return serie

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Crear una serie temporal de ejemplo
    data = {'ventas': [100, 120, 130, 125, 140, 150, 160]}
    df = pd.DataFrame(data)

    # Verificar estacionariedad
    df['ventas'] = verificar_estacionariedad(df['ventas'])

    # Entrenar modelo ARIMA
    modelo_arima = entrenar_modelo_arima(df['ventas'])
    print("Resumen del modelo ARIMA:")
    print(modelo_arima.summary())

    # Graficar la serie temporal diferenciada
    df['ventas'].plot(title="Serie Temporal Diferenciada")
    plt.show()

    # Graficar los pronósticos del modelo ARIMA
    forecast = modelo_arima.get_forecast(steps=3)
    forecast_index = pd.date_range(start=df.index[-1], periods=4, freq='M')[1:]
    forecast_mean = forecast.predicted_mean
    plt.plot(df['ventas'], label="Datos originales")
    plt.plot(forecast_index, forecast_mean, label="Pronóstico", color='red')
    plt.legend()
    plt.show()

    # Entrenar modelo Ridge
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([100, 120, 130, 125, 140, 150])
    modelo_ridge_entrenado = modelo_ridge(X, y)
    print("Modelo Ridge entrenado:")
    print(modelo_ridge_entrenado)