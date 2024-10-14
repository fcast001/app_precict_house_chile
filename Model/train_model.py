import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Cargar el dataset
real_estate_data = pd.read_csv("Model/Data/Real_Estate.csv")
real_estate_data.rename(columns={'Distance to the nearest MRT station': 'distance_to_mrt', 
                                 'Number of convenience stores': 'num_convenience_stores',
                                 'Latitude': 'latitude', 
                                 'Longitude': 'longitude'}, inplace=True)

# Renombrar las columnas
real_estate_data.rename(columns={
    'Distance to the nearest MRT station': 'distance_to_mrt', 
    'Number of convenience stores': 'num_convenience_stores',
    'Latitude': 'latitude', 
    'Longitude': 'longitude'
}, inplace=True)

# Seleccionar características y variable objetivo
features = ['distance_to_mrt', 'num_convenience_stores', 'latitude', 'longitude']
target = 'House price of unit area'

X = real_estate_data[features]
y = real_estate_data[target]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Guardar el modelo entrenado en un archivo pickle
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)


