#Cargar el archivo pacientes.csv, escalar las columnas edad y colesterol, y luego convertirlo al formato solicitado.
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import math
import joblib
import matplotlib.pyplot as plt


# 1. Cargar el archivo CSV
df = pd.read_csv('pacientes.csv')
#Realizar un grafico de dispersión entre edad y colesterol, coloreando los puntos según el problema_cardiaco
#imprimir el maximo y el minimo de la edad y del colesterol

imputer = SimpleImputer(strategy="mean")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("Edad - Max:", df['edad'].max(), "Min:", df['edad'].min())
print("Colesterol - Max:", df['colesterol'].max(), "Min:", df['colesterol'].min())
plt.scatter(df['edad'], df['colesterol'], c=df['problema_cardiaco'], cmap='viridis')
plt.xlabel('Edad')
plt.ylabel('Colesterol')
plt.title('Dispersión entre Edad y Colesterol')
plt.colorbar(label='Problema Cardiaco')
plt.show()
                    
#Caargar el modelo de escalado de joblib del archivo scaler.joblib
scaler = joblib.load('scaler.jb')
#Con esta función ce una red neuronal se va a predecir si va a tener problemas cardiacos o no, se va a usar el modelo de escalado para escalar las columnas edad y colesterol 
def predecir_problema_cardiaco(edad, colesterol):
    # Escalar las columnas X1,x2 usando el modelo de escalado cargado
    datos_transformados = scaler.transform([[edad, colesterol]])
    X1 = datos_transformados[0, 0]*2
    X2 = datos_transformados[0, 1]*2
    print("Datos escalados - Edad:", X1, "Colesterol:", X2)
    # Convertir al formato solicitado
    """Compute a forward pass of the network."""
    a1 = 1 / (1 + math.exp(-(1.5 + (-4.8 * X1) + (7.3 * X2))))
    a2 = 1 / (1 + math.exp(-(3.5 + (1.7 * X1) + (-6.7 * X2))))
    a3 = 1 / (1 + math.exp(-(-5.2 + (0.81 * X1) + (-1.4 * X2))))
    a4 = 1 / (1 + math.exp(-(-1.6 + (0.43 * X1) + (2.2 * X2))))
    a5 = 1 / (1 + math.exp(-(-1.3 + (-0.66 * X1) + (-0.43 * X2))))
    a6 = 1 / (1 + math.exp(-(2.7 + (1.1 * X1) + (0.062 * X2))))
    a7 = 1 / (1 + math.exp(-(0.61 + (-1.1 * a1) + (-1.9 * a2) + (-0.95 * a3) + (0.76 * a4) + (-0.81 * a5) + (0.88 * a6))))
    a8 = 1 / (1 + math.exp(-(1.5 + (-2.1 * a1) + (-3.9 * a2) + (-1.8 * a3) + (1.6 * a4) + (-0.98 * a5) + (1.8 * a6))))
    a9 = 1 / (1 + math.exp(-(-2.5 + (3.2 * a1) + (3.7 * a2) + (2.3 * a3) + (-1.4 * a4) + (0.80 * a5) + (-2.5 * a6))))
    a10 = 1 / (1 + math.exp(-(0.056 + (4.1 * a1) + (1.4 * a2) + (0.79 * a3) + (-5.0 * a4) + (1.7 * a5) + (0.11 * a6))))
    a11 = 1 / (1 + math.exp(-(-0.32 + (0.81 * a7) + (1.4 * a8) + (-2.3 * a9) + (-6.8 * a10))))
    a12 = 1 / (1 + math.exp(-(-2.3 + (-2.5 * a7) + (-4.9 * a8) + (4.9 * a9) + (1.2 * a10))))
    a13 = math.tanh(0.44 + (4.4 * a11) + (-3.8 * a12))
    return a13

# Crear un DataFrame con los datos de entrada
edad=int(input("Ingrese la edad: "))
colesterol=int(input("Ingrese el colesterol: "))
resultado = predecir_problema_cardiaco(edad, colesterol)
# Clasificación final
clase = 1 if resultado >= 0 else -1

print("Resultado de la predicción:", resultado)
print("Clase predicha:", clase)
if clase == 1:
    print("El paciente Si presenta problema cardíaco")
else:
    print("El paciente No presenta problema cardíaco")
