import pandas as pd
import streamlit as st
import pickle

# Cargar el modelo
with open("Model/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Cargar el dataset
real_estate_data = pd.read_csv("Data/Real_Estate.csv")  # Reemplaza la ruta según sea necesario

# Cambiar el nombre de las columnas para que coincidan con las que usas
real_estate_data.rename(columns={
    'Distance to the nearest MRT station': 'distance_to_mrt',
    'Number of convenience stores': 'num_convenience_stores',
    'Latitude': 'latitude',
    'Longitude': 'longitude'
}, inplace=True)

# Título de la aplicación
st.title("Predicción de Precios de Bienes Raíces")

# Crear dos columnas
col1, col2 = st.columns(2)

# Inputs en la primera columna
with col1:
    distance_to_mrt = st.number_input('Distancia a la estación de MRT (metros)', min_value=0)
    num_convenience_stores = st.number_input('Número de tiendas de conveniencia', min_value=0)
    
# Inputs en la segunda columna
with col2:
    latitude = st.number_input('Latitud', format="%.6f")
    longitude = st.number_input('Longitud', format="%.6f")

# Botón para predecir el precio
if st.button('Predecir Precio'):
    # Preparar el vector de características
    features = pd.DataFrame([[distance_to_mrt, num_convenience_stores, latitude, longitude]],
                            columns=['distance_to_mrt', 'num_convenience_stores', 'latitude', 'longitude'])
    
    # Predecir
    prediction = model.predict(features)[0]
    st.success(f'Precio predicho por unidad de área: {prediction:.2f}')
