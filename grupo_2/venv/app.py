import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Función para preprocesar los datos de entrada
def preprocess_data(data):

    feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFlaggedFraud']
    
    # Mapear los tipos de transacción a valores numéricos
    data['type'] = data['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})
    
    # Escalamiento de características
    data_scaled = scaler.transform(data[feature_names])
    
    return data_scaled

# Aplicación en Streamlit
def main():
    st.title("Aplicación para Detección de Transacciones Fraudulentas")

    # Obtener datos de entrada del usuario
    st.header("Ingrese los Detalles de la Transacción:")
    step = st.number_input("Paso", min_value=1)
    type_val = st.selectbox("Tipo de Transacción", ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'])
    amount = st.number_input("Monto")
    oldbalanceOrg = st.number_input("Saldo Anterior del Origen")
    oldbalanceDest = st.number_input("Saldo Anterior del Destino")
    isFlaggedFraud = st.checkbox("Marcado como Fraude")

    # Botón de envío
    if st.button("Enviar"):
        # Crear un DataFrame con los datos ingresados por el usuario
        user_data = pd.DataFrame({
            'step': [step],
            'type': [type_val],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'oldbalanceDest': [oldbalanceDest],
            'isFlaggedFraud': [isFlaggedFraud]
        })

        # Preprocesar los datos ingresados por el usuario
        user_data_scaled = preprocess_data(user_data)

        # Realizar una predicción
        prediction = model.predict(user_data_scaled)

        # Mostrar el resultado
        st.header("Predicción:")
        if prediction[0] == 1:
            st.error("¡Esta transacción se predice como Fraude!")
        else:
            st.success("Esta transacción se predice como No Fraudulenta.")

if __name__ == '__main__':
    main()
