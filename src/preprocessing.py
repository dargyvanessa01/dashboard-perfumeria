import pandas as pd

def cargar_datos(ruta_csv):
    """
    Carga el archivo CSV y realiza la conversión de la columna de fechas.
    """
    df = pd.read_csv(ruta_csv, parse_dates=['Mes'])
    return df

def crear_features(df):
    """
    Genera nuevas características a partir de los datos originales.
    """
    df = df.copy()
    
    # Lag features
    df['ventas_1oz_lag1'] = df['1 oz'].shift(1)
    df['ventas_2oz_rolling3'] = df['2 oz'].rolling(window=3).mean()
    
    # Variables categóricas
    df['trimestre'] = df['Mes'].dt.quarter
    
    # Eliminación de valores NaN generados por los shifts y rollings
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    ruta_csv = "c:/Users/vanes/OneDrive/Documentos/Proyecto_Final/data/Proyecto_final.csv"
    df = cargar_datos(ruta_csv)
    df = crear_features(df)
    print(df.head())



