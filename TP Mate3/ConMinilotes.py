# CON MINILOTES

# LLamado de las librerias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Declaramos el dataFrame como all_data
all_data = pd.read_csv(r'C:\Users\Ke Ba\Desktop\TP Mate3\df_limpio.csv')

# Separamos los datos a tener en cuenta, por un lado los datos de entrada, por el otro los de salida ("class")
all_inputs = all_data.iloc[:, 0:8].values
all_outputs = all_data.iloc[:, -1].values  # Asegúrate de que esta columna tenga 3 clases (0, 1, 2)

# Declaramos la funcion a utilizar para normalizar los datos y le damos play
def normalizador(X):
    # Calcular la media y la desviación estándar
    promedio = np.mean(X, axis=0)  # Media de cada característica
    desvEst = np.std(X, axis=0)    # Desviación estándar de cada característica

    # Aplicar la transformación
    datoNormalizado = (X - promedio) / desvEst
    return datoNormalizado

all_inputs = normalizador(all_inputs)

### Convertir las salidas a formato one-hot

# Declaramos la cantidad de respuestas posibles (por no ser binario)
respPosibles = 3

# Con nume.eye(respPosibles) generamos una matriz identidad 3x3 que va a reemplazar a los valores (0,1,2) originales
# Al combinarla con all_outputs vinculamos cada fila de la MI con cada valor
# La primer fila sería 0, la segunda 1, y la tercera 2
# Hacemos esto para poder trabajar las 3 posibilidades dentro de la red.
y_matriz = np.eye(respPosibles)[all_outputs]

# Dividir en un conjunto de entrenamiento y uno de prueba
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, y_matriz, test_size=1/3)

n = X_train.shape[0]  # número de registros de entrenamiento

# Funciones de activación
relu = lambda x: np.maximum(x, 0)

# Función softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # estabilización numérica
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

# Construir una red neuronal con pesos y sesgos iniciados aleatoriamente
# Entre las pruebas que hicimos, con 10 neuronas se obtuvieron muy buenos resultados

np.random.seed(1)
w_hidden = (np.random.rand(10, 8) * 2) - 1
w_output = (np.random.rand(3, 10) * 2) - 1

b_hidden = (np.random.rand(10, 1) * 2) - 1
b_output = (np.random.rand(3, 1) * 2) - 1

# Función que corre la red neuronal con los datos de entrada para predecir la salida
def forward_prop(X):
    Z1 = w_hidden @ X.T + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = softmax(Z2)  # Cambiamos a softmax para la salida
    return Z1, A1, Z2, A2

# Cálculo de precisión
def precision(X, Y):
    test_predictions = forward_prop(X)[3]  # solo nos interesa A2
    predicted_classes = np.argmax(test_predictions, axis=0)  # obtener clase con mayor probabilidad
    true_classes = np.argmax(Y, axis=1)  # etiquetas verdaderas
    accuracy = np.mean(predicted_classes == true_classes)  # porcentaje de aciertos
    print("Porcentaje de aciertos: ", (accuracy * 100).round(2))

print('Pre entrenamiento: \n')
print('Test')
precision(X_test, Y_test)
print('Train')
precision(X_train, Y_train)

# DE ACA PARA ABAJO IMPORTANTE QUE NO LO ESCRIBE COMO APRENDIMOS
# TENER EN CUENTA VER SI SE PUEDE HACER ALGO SIMILAR A LO QUE
# PUSE EN EL TP FINAL DE 2 VALORES, QUE ES COMO LO RESOLVIO
# EL PROFE EN EL UNIDAD 6

# Devuelve pendientes para pesos y sesgos usando la regla de la cadena
def backward_prop(Z1, A1, Z2, A2, X, Y):
    m = X.shape[0]  # número de ejemplos
    dC_dA2 = A2 - Y.T  # error de salida
    dC_dW2 = (dC_dA2 @ A1.T) / m  # (3, m) @ (m, 10) = (3, 10)
    dC_dB2 = np.sum(dC_dA2, axis=1, keepdims=True) / m  # (3, m) -> (3, 1)

    dA1_dZ1 = (Z1 > 0).astype(float)  # derivada de ReLU
    dC_dA1 = w_output.T @ dC_dA2  # (10, 3) @ (3, m) = (10, m)
    dC_dZ1 = dC_dA1 * dA1_dZ1  # (10, m)

    dC_dW1 = (dC_dZ1 @ X) / m  # (10, m) @ (m, 8) = (10, 8)
    dC_dB1 = np.sum(dC_dZ1, axis=1, keepdims=True) / m  # (10, m) -> (10, 1)

    return dC_dW1, dC_dB1, dC_dW2, dC_dB2

# La tasa de aprendizaje
L = 0.01

# Ejecutar descenso de gradiente
tamanioLote = 32  # Minilotes
num_epochs = 50_000  # Aumentar el número de épocas

for i in tqdm(range(num_epochs)):
    idx = np.random.choice(n, tamanioLote, replace=False)

    X_sample = X_train[idx]
    Y_sample = Y_train[idx]

    Z1, A1, Z2, A2 = forward_prop(X_sample)

    dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

    w_hidden -= L * dW1
    b_hidden -= L * dB1
    w_output -= L * dW2
    b_output -= L * dB2

# Cálculo de precisión
print('Post entrenamiento: \n')
print('Test')
precision(X_test, Y_test)
print('Train')
precision(X_train, Y_train)
