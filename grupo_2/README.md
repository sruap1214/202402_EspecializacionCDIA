# **Grupo #2 - Transacciones Fraudulentas**

- **Participantes**: Luissa Maria Valencia López, Cristian Hernández Vanegas, Hector Dario Betancur

# **Predicción de Transacciones Fraudulentas**

## **Resumen**
Este repositorio contiene un proyecto de **Detección de Fraude** que se enfoca en predecir si una transacción financiera es fraudulenta o no. Utiliza modelos de aprendizaje automático, específicamente el **Clasificador Random Forest**.  
El modelo entrenado se guarda para su uso futuro, y se incluye una aplicación **Streamlit** para facilitar la interacción con el modelo.

---

## **Datos**
El conjunto de datos utilizado para el entrenamiento y análisis se llama **Fraud.csv**.  
**Características:**
- **Tamaño:** Aproximadamente 6.5 millones de filas.
- **Columnas:** 10 características relacionadas con transacciones financieras.

---

## **Estructura del Proyecto**
La carpeta del proyecto incluye los siguientes archivos:

- **`model.sav`**: Modelo de **Random Forest Classifier** guardado.
- **`scaler.sav`**: Objeto **StandardScaler** guardado, utilizado para la escala de características.
- **`requirements.txt`**: Dependencias necesarias para el proyecto.
- **`app.py`**: Aplicación **Streamlit** que permite ingresar detalles de transacciones y recibir predicciones.
- **`fraud-detection.ipynb`**: Análisis exploratorio de datos (EDA), procesamiento de datos, entrenamiento del modelo y evaluación.

---

## **Cómo Usar**

### **Instalación**
Sigue estos pasos para configurar el proyecto en tu máquina local:

1. **Instala las dependencias necesarias:**
    ```bash
    pip install -r requirements.txt
## **Uso**
### **Aplicación Streamlit**
1. Asegúrate de haber instalado todas las dependencias necesarias.
2. Ejecuta la aplicación Streamlit:
    ```bash
    streamlit run app.py
3. Abre tu navegador web para interactuar con la aplicación de Detección de Fraude.
Despliegue del Modelo
Si deseas usar el modelo entrenado programáticamente, puedes hacerlo siguiendo este ejemplo en Python:

python

    import pickle
    import numpy as np
    
    # Cargar el modelo y el escalador guardados
    model = pickle.load(open('model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    
    # Crear un array de entrada con los detalles de la transacción
    input_array = np.array([[228, 5, 117563.11, 0.0, 208908.41, 0]])
    
    # Escalar el array de entrada
    input_array_scaled = scaler.transform(input_array)
    
    # Hacer una predicción
    prediction = model.predict(input_array_scaled)
    
    print("Predicción:", prediction)
    Nota: Ajusta el array de entrada según los detalles específicos de la transacción que deseas predecir.