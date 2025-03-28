# Proyecto Final: Análisis de Ventas con Dashboard Interactivo

## Instalación
1. Clonar repositorio: `git clone [URL]`
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar dashboard: `python src/app.py`

## Características Clave
- ✅ Pronósticos ARIMA con intervalos de confianza
- ✅ Regularización Ridge con ajuste interactivo de α
- ✅ Análisis de incertidumbre (coeficiente de variación, error estándar)
- ✅ Visualizaciones interactivas con Plotly/Dash

## Estructura del Código
| Archivo          | Descripción                             |
|------------------|-----------------------------------------|
| `src/app.py`     | Dashboard interactivo                   |
| `src/model.py`   | Modelos predictivos y estadísticos      |
| `data/`          | Datos históricos de ventas              |