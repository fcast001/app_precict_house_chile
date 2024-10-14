import streamlit as st
import pandas as pd
import pickle

# Cargar el modelo entrenado
with open('Model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Crear la interfaz de Streamlit
st.title('Real Estate Price Prediction')

# Crear dos columnas para las entradas
col1, col2 = st.columns(2)

# Entradas en la primera columna
with col1:
    distance_to_mrt = st.number_input('Distance to MRT Station (meters)', min_value=0.0, step=1.0)
    num_convenience_stores = st.number_input('Number of Convenience Stores', min_value=0, step=1)

# Entradas en la segunda columna
with col2:
    latitude = st.number_input('Latitude', min_value=0.0, step=0.01)
    longitude = st.number_input('Longitude', min_value=0.0, step=0.01)

# Botón para hacer la predicción
if st.button('Predict'):
    # Crear el dataframe con las características de entrada
    input_data = pd.DataFrame([[distance_to_mrt, num_convenience_stores, latitude, longitude]], 
                              columns=['distance_to_mrt', 'num_convenience_stores', 'latitude', 'longitude'])
    
    # Hacer la predicción
    prediction = model.predict(input_data)[0]
    
    # Mostrar el resultado
    st.success(f'The predicted house price of unit area is: {prediction:.2f}')
